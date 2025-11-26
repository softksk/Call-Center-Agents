import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import openai
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from ..core.config import settings

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """
    Состояние агента для передачи между узлами в LangGraph.

    Центральная модель данных, используемая для передачи информации
    между различными агентами в multi-agent системе. Содержит всю
    необходимую контекстную информацию о диалоге и его состоянии.

    Attributes
    ----------
        dialog_id: Уникальный идентификатор текущего диалога.
        current_agent: Имя агента, обрабатывающего сообщение.
        user_message: Текущее сообщение от пользователя.
        message_history: Полная история сообщений в диалоге.
        should_continue: Флаг продолжения обработки в workflow.
        next_agent: Агент для передачи управления.

    """

    dialog_id: str = Field(..., description="Идентификатор диалога")
    current_agent: str = Field(..., description="Текущий активный агент")
    previous_agent: Optional[str] = Field(None, description="Предыдущий агент")
    user_message: str = Field(..., description="Текущее сообщение пользователя")
    agent_response: Optional[str] = Field(None, description="Ответ агента")
    message_history: List[BaseMessage] = Field(
        default_factory=list, description="История сообщений диалога"
    )
    user_intent: Optional[str] = Field(None, description="Намерение пользователя")
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict, description="Извлеченные сущности"
    )
    escalation_reason: Optional[str] = Field(None, description="Причина эскалации")
    conversation_summary: Optional[str] = Field(None, description="Краткое содержание разговора")
    should_continue: bool = Field(True, description="Продолжать ли обработку")
    next_agent: Optional[str] = Field(None, description="Следующий агент для передачи")
    handoff_reason: Optional[str] = Field(None, description="Причина передачи управления")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")


class BaseAgent(ABC):
    """
    Базовый класс для всех агентов системы колл-центра.

    Абстрактный базовый класс, определяющий общий интерфейс и функциональность
    для всех специализированных агентов. Каждый агент имеет доступ к LLM
    и может обрабатывать сообщения пользователей с возможностью передачи
    управления другим агентам.

    Attributes
    ----------
        name: Уникальное имя агента в системе.
        system_prompt: Системная инструкция для настройки поведения LLM.
        _llm: Экземпляр языковой модели для генерации ответов.

    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> None:
        """
        Инициализация базового агента.

        Args:
        ----
            name: Имя агента для идентификации
            system_prompt: Системный промпт, определяющий роль агента
            model_name: Название модели OpenAI для использования
            temperature: Температура для генерации ответов (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе

        """
        self.name = name
        self.system_prompt = system_prompt

        # Инициализируем OpenAI клиент
        self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Агент {name} инициализирован с OpenAI {model_name}")

    async def process(self, state: AgentState) -> str:
        """
        Обработка входящего сообщения через OpenAI API.

        Args:
        ----
            state: Текущее состояние агента с сообщением пользователя

        Returns:
        -------
            Ответ агента на сообщение пользователя

        """
        try:
            # Формируем сообщения для OpenAI API, начиная с системного промпта
            messages = [{"role": "system", "content": self.system_prompt}]

            # Добавляем историю сообщений для контекста диалога
            if state.message_history:
                for msg in state.message_history:
                    if hasattr(msg, "type"):
                        # LangChain message format (HumanMessage/AIMessage)
                        role = "user" if msg.type == "human" else "assistant"
                        messages.append({"role": role, "content": msg.content})
                    elif isinstance(msg, dict):
                        # Dict format с ролями
                        messages.append(msg)

            # Добавляем текущее сообщение пользователя
            messages.append({"role": "user", "content": state.user_message})

            # Получаем ответ от OpenAI
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка в агенте {self.name}: {type(e).__name__}: {e}")
            logger.debug(f"Детали ошибки в агенте {self.name}", exc_info=True)
            return "Извините, произошла техническая ошибка. Попробуйте переформулировать запрос."

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Возвращает список возможностей агента.

        Абстрактный метод, который должен быть реализован каждым
        конкретным агентом для описания своих основных функций
        и специализаций.

        Returns
        -------
            Список строк с описанием возможностей агента

        """
        pass
