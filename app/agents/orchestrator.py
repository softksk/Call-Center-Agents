import logging
from typing import Any, Dict, List, Optional

import openai
from langgraph.graph import END, StateGraph

from ..core.config import settings
from .base import AgentState
from .prompts import format_handoff_prompt, format_routing_prompt
from .router import RouterAgent
from .sales import SalesAgent
from .supervisor import SupervisorAgent
from .tech_support import TechSupportAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Оркестратор агентов на основе LangGraph для управления мультиагентными диалогами.

    Координирует взаимодействие между специализированными агентами колл-центра,
    используя граф состояний LangGraph и интеллектуальную маршрутизацию на основе
    языковых моделей. Обеспечивает seamless переходы между агентами и контекстуальное
    принятие решений о направлении диалога.

    Класс управляет четырьмя типами агентов:
    - Router: Первичная маршрутизация и классификация запросов
    - TechSupport: Техническая поддержка и диагностика
    - Sales: Продажи и консультации по продуктам
    - Supervisor: Эскалация и разрешение конфликтов

    Attributes
    ----------
        openai_client: Асинхронный клиент OpenAI для принятия решений о маршрутизации
        agents: Словарь инициализированных агентов системы
        graph: Скомпилированный граф состояний LangGraph для обработки диалогов

    """

    def __init__(self) -> None:
        """
        Инициализация оркестратора с OpenAI клиентом и агентами.

        Создает все необходимые компоненты для функционирования системы:
        асинхронный клиент OpenAI для принятия решений о маршрутизации,
        экземпляры всех специализированных агентов и граф состояний
        для управления потоком диалога.

        Raises
        ------
            Exception: Если не удается инициализировать OpenAI клиент или
                построить граф состояний

        """
        # Инициализируем OpenAI клиент для маршрутизации
        self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        # Инициализируем агентов
        self.agents = {
            "router": RouterAgent(),
            "tech_support": TechSupportAgent(),
            "sales": SalesAgent(),
            "supervisor": SupervisorAgent(),
        }

        # Строим граф
        self.graph = self._build_agent_graph()

    def _build_agent_graph(self) -> Any:
        """
        Построение LangGraph с LLM-управляемой маршрутизацией.

        Создает сложный граф состояний для управления потоком диалога между
        специализированными агентами. Каждый переход в графе определяется
        через интеллектуальный анализ контекста разговора с помощью языковой
        модели, обеспечивая естественное направление пользователей к нужным
        агентам без жестких правил.

        Граф включает:
        - Узлы для каждого типа агента (router, tech_support, sales, supervisor)
        - Условные переходы на основе LLM-анализа контекста
        - Автоматическое завершение диалогов супервизором
        - Защитные механизмы при ошибках маршрутизации

        Returns
        -------
            Скомпилированный граф LangGraph, готовый для обработки диалогов

        Raises
        ------
            Exception: Если не удается создать или скомпилировать граф состояний

        """
        # Создаем граф состояний
        workflow = StateGraph(AgentState)

        # Добавляем узлы для каждого агента
        workflow.add_node("router", self._router_node)
        workflow.add_node("tech_support", self._tech_support_node)
        workflow.add_node("sales", self._sales_node)
        workflow.add_node("supervisor", self._supervisor_node)

        # Определяем точку входа - всегда начинаем с роутера
        workflow.set_entry_point("router")

        # Добавляем LLM-управляемые условные переходы
        workflow.add_conditional_edges(
            "router",
            self._llm_route_decision,
            {
                "tech_support": "tech_support",
                "sales": "sales",
                "supervisor": "supervisor",
                "end": END,
            },
        )

        # Каждый агент может передать управление другому агенту или завершить
        for agent in ["tech_support", "sales"]:
            workflow.add_conditional_edges(
                agent,
                self._llm_continue_or_handoff,
                {
                    "tech_support": "tech_support",
                    "sales": "sales",
                    "supervisor": "supervisor",
                    "end": END,
                },
            )

        # Супервизор всегда завершает диалог
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_handoff,
            {
                "end": END,
            },
        )

        return workflow.compile(checkpointer=None, debug=False)

    async def _llm_route_decision(self, state: AgentState) -> str:
        """
        Интеллектуальное принятие решения о первичной маршрутизации пользователя.

        Использует языковую модель для анализа входящего сообщения пользователя
        и определения наиболее подходящего агента для обработки запроса.
        Применяет few-shot промптинг с примерами для повышения точности
        классификации и обеспечения консистентной маршрутизации.

        Args:
        ----
            state: Текущее состояние диалога с сообщением пользователя

        Returns:
        -------
            Название агента для обработки запроса: 'tech_support', 'sales',
            'supervisor' или 'end' для завершения диалога

        Raises:
        ------
            Exception: При ошибках API OpenAI возвращает 'supervisor' как fallback

        """
        try:
            routing_prompt = format_routing_prompt(state.user_message)

            # Используем OpenAI API напрямую
            response = await self.openai_client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "system", "content": routing_prompt}],
                temperature=0.1,
                max_tokens=50,
            )

            decision = response.choices[0].message.content.strip().lower()

            # Валидация решения
            valid_routes = ["tech_support", "sales", "supervisor", "end"]
            if decision not in valid_routes:
                logger.warning(f"Invalid routing decision: {decision}, defaulting to supervisor")
                # Простой fallback - направляем к супервизору для обработки неясных случаев
                return "supervisor"

            logger.info(f"LLM routing decision: {state.user_message[:50]}... -> {decision}")
            return decision

        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            # При ошибке направляем к супервизору для безопасности
            return "supervisor"

    async def _llm_continue_or_handoff(self, state: AgentState) -> str:
        """
        Интеллектуальное решение о продолжении или передаче управления между агентами.

        Анализирует контекст диалога после ответа агента и принимает решение
        о дальнейшем направлении разговора. Использует полностью LLM-управляемую
        логику без жестких правил, обеспечивая естественные переходы между
        агентами на основе потребностей пользователя и качества обслуживания.

        Args:
        ----
            state: Текущее состояние диалога с сообщением пользователя,
                ответом агента и информацией о текущем агенте

        Returns:
        -------
            Решение о следующем шаге: 'tech_support', 'sales', 'supervisor'
            для передачи другому агенту, или 'end' для завершения диалога

        Raises:
        ------
            Exception: При ошибках API возвращает 'supervisor' как безопасный fallback

        """
        try:
            # Используем LLM для принятия решения о переходе
            handoff_prompt = format_handoff_prompt(
                user_message=state.user_message,
                agent_response=state.agent_response or "",
                current_agent=state.current_agent,
            )

            # Используем OpenAI API напрямую
            response = await self.openai_client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "system", "content": handoff_prompt}],
                temperature=0.1,
                max_tokens=50,
            )

            decision = response.choices[0].message.content.strip().lower()

            # Валидация решения
            valid_handoffs = ["tech_support", "sales", "supervisor", "end"]
            if decision not in valid_handoffs:
                logger.warning(f"Invalid handoff decision: {decision}, defaulting to end")
                return "end"

            logger.info(f"LLM handoff decision: {state.current_agent} -> {decision}")
            return decision

        except Exception as e:
            logger.error(f"Error in LLM handoff: {e}")
            # При ошибке направляем к супервизору для безопасности
            return "supervisor"

    async def _supervisor_handoff(self, state: AgentState) -> str:
        """
        Завершение диалога супервизором.

        Супервизор всегда завершает диалог после обработки, так как является
        конечной точкой эскалации в системе. Этот метод обеспечивает
        корректное закрытие проблемных или конфликтных диалогов.

        Args:
        ----
            state: Текущее состояние диалога (не используется, но требуется
                для совместимости с интерфейсом LangGraph)

        Returns:
        -------
            Всегда возвращает 'end' для завершения диалога

        """
        logger.info("Supervisor completing dialog")
        return "end"

    async def _router_node(self, state: AgentState) -> AgentState:
        """
        Узел роутера в графе состояний LangGraph.

        Обрабатывает входящее сообщение пользователя через агента-роутера,
        который выполняет первичную классификацию запроса и подготавливает
        состояние для последующей маршрутизации к специализированным агентам.

        Args:
        ----
            state: Состояние диалога с пользовательским сообщением

        Returns:
        -------
            Обновленное состояние диалога с ответом роутера и установленным
            текущим агентом 'router'

        """
        logger.info(f"Router processing: {state.user_message[:50]}...")
        logger.debug(f"Router has message_history with {len(state.message_history)} messages")

        # Если state - словарь, преобразуем в AgentState
        if isinstance(state, dict):
            state = AgentState(**state)

        response = await self.agents["router"].process(state)

        # Обновляем состояние
        state.agent_response = response
        state.current_agent = "router"
        state.next_agent = ""  # Будет определено LLM

        from langchain_core.messages import AIMessage, HumanMessage

        state.message_history = [*state.message_history, HumanMessage(content=state.user_message), AIMessage(content=response)]

        return state

    async def _tech_support_node(self, state: AgentState) -> AgentState:
        """Узел техподдержки в графе."""
        logger.info(f"Tech support processing: {state.user_message[:50]}...")
        logger.debug(f"Tech support has message_history with {len(state.message_history)} messages")

        # Если state - словарь, преобразуем в AgentState
        if isinstance(state, dict):
            state = AgentState(**state)

        response = await self.agents["tech_support"].process(state)

        # Обновляем состояние
        state.agent_response = response
        state.current_agent = "tech_support"
        state.next_agent = ""

        # ВАЖНО: Добавляем текущий обмен в историю для следующих агентов
        from langchain_core.messages import AIMessage, HumanMessage

        state.message_history = [
            *state.message_history,
            HumanMessage(content=state.user_message),
            AIMessage(content=response),
        ]

        # Проверяем, нужно ли немедленно завершить диалог
        message_lower = state.user_message.lower()
        if any(
            phrase in message_lower
            for phrase in ["решил сам", "всё решил", "спасибо", "всё нормально"]
        ):
            state.should_continue = False
            state.next_agent = "end"

        return state

    async def _sales_node(self, state: AgentState) -> AgentState:
        """Узел продаж в графе."""
        logger.info(f"Sales processing: {state.user_message[:50]}...")
        logger.debug(f"Sales has message_history with {len(state.message_history)} messages")

        # Если state - словарь, преобразуем в AgentState
        if isinstance(state, dict):
            state = AgentState(**state)

        response = await self.agents["sales"].process(state)

        # Обновляем состояние
        state.agent_response = response
        state.current_agent = "sales"
        state.next_agent = ""

        # ВАЖНО: Добавляем текущий обмен в историю для следующих агентов
        from langchain_core.messages import AIMessage, HumanMessage

        state.message_history = [
            *state.message_history,
            HumanMessage(content=state.user_message),
            AIMessage(content=response),
        ]

        # Проверяем, нужно ли немедленно завершить диалог
        message_lower = state.user_message.lower()
        if any(
            phrase in message_lower for phrase in ["спасибо", "всё понятно", "не нужно", "отменяю"]
        ):
            state.should_continue = False
            state.next_agent = "end"

        return state

    async def _supervisor_node(self, state: AgentState) -> AgentState:
        """Узел супервизора в графе."""
        logger.info(f"Supervisor processing: {state.user_message[:50]}...")

        # Если state - словарь, преобразуем в AgentState
        if isinstance(state, dict):
            state = AgentState(**state)

        response = await self.agents["supervisor"].process(state)

        # Обновляем состояние
        state.agent_response = response
        state.current_agent = "supervisor"
        state.next_agent = "end"  # Супервизор всегда завершает диалог
        state.should_continue = False  # Принудительно завершаем

        return state

    async def process_message(
        self,
        message: str,
        dialog_id: str,
        conversation_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Основной метод обработки сообщения пользователя через LangGraph.

        Координирует весь процесс обработки пользовательского запроса через
        граф состояний, включая маршрутизацию, обработку специализированными
        агентами и принятие решений о передачах. Обеспечивает отказоустойчивость
        и корректную обработку ошибок на всех этапах.

        Args:
        ----
            message: Текст сообщения пользователя для обработки
            dialog_id: Уникальный идентификатор диалога для отслеживания
            conversation_history: Опциональная история предыдущих сообщений
                в диалоге для контекста

        Returns:
        -------
            Словарь с результатом обработки, содержащий:
                - agent_response: Ответ агента пользователю
                - current_agent: Название агента, обработавшего запрос
                - dialog_id: Идентификатор диалога
                - metadata: Дополнительная информация о обработке

        Raises:
        ------
            Exception: При критических ошибках возвращает стандартное
                сообщение об ошибке вместо исключения

        """
        try:
            # Извлекаем историю сообщений из conversation_history, если она есть
            message_history = []
            if conversation_history and "messages" in conversation_history:
                message_history = conversation_history["messages"]

            # Создаем начальное состояние с историей сообщений
            initial_state = AgentState(
                dialog_id=dialog_id,
                current_agent="router",
                user_message=message,
                agent_response="",
                message_history=message_history,  # Передаем историю из conversation_history
                next_agent="router",
                metadata={},
            )

            # Запускаем граф с ограничением рекурсии
            logger.info(f"Starting LangGraph processing for dialog {dialog_id}")
            final_state_dict = await self.graph.ainvoke(
                initial_state, config={"recursion_limit": 10}  # Увеличиваем до 10 итераций
            )

            # Преобразуем результат в AgentState, если это словарь
            if isinstance(final_state_dict, dict):
                final_state = AgentState(**final_state_dict)
            else:
                final_state = final_state_dict

            # Формируем результат
            result = {
                "agent_response": final_state.agent_response,
                "current_agent": final_state.current_agent,
                "dialog_id": dialog_id,
                "metadata": final_state.metadata,
            }

            logger.info(
                f"LangGraph completed: {final_state.current_agent} -> {len(final_state.agent_response)} chars"
            )
            return result

        except Exception as e:
            logger.error(f"Error in LangGraph processing: {e}", exc_info=True)
            # При ошибке пытаемся вернуть роутер вместо error, чтобы не терять контекст
            return {
                "agent_response": "Извините, произошла техническая ошибка. Попробуйте переформулировать запрос, и я постараюсь помочь.",
                "current_agent": "router",  # Изменено с "error" на "router"
                "dialog_id": dialog_id,
                "metadata": {"error": str(e), "error_type": type(e).__name__},
            }

    async def process_dialog_turn(
        self,
        dialog_id: str,
        user_message: str,
        message_history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Обработка диалогового хода для совместимости с API.

        Args:
        ----
            dialog_id: Идентификатор диалога
            user_message: Сообщение пользователя
            message_history: История сообщений

        Returns:
        -------
            Результат обработки с ответом агента

        """
        # Преобразуем в формат, ожидаемый process_message
        conversation_history = {"messages": message_history or []}
        return await self.process_message(user_message, dialog_id, conversation_history)

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Получение информации о конфигурации агентов системы.

        Предоставляет полную информацию о доступных агентах, их возможностях
        и структуре графа состояний для мониторинга, отладки и документации
        системы.

        Returns
        -------
            Словарь с информацией о системе агентов, содержащий:
                - agents: Список названий доступных агентов
                - capabilities: Возможности каждого агента
                - graph_nodes: Узлы графа состояний LangGraph

        """
        return {
            "agents": list(self.agents.keys()),
            "capabilities": {name: agent.get_capabilities() for name, agent in self.agents.items()},
            "graph_nodes": ["router", "tech_support", "sales", "supervisor"],
        }
