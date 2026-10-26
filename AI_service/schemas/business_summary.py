from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from AI_service.schemas.chat import BusinessContext, TaskContext


SummaryText = Annotated[str, Field(max_length=1000)]


class BusinessSummaryRequest(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    customer_identifier: str | None = Field(default=None, max_length=200)
    business_context: BusinessContext | None = None
    task_context: TaskContext | None = None

    @field_validator("customer_identifier")
    @classmethod
    def normalize_customer_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to.")
        return self


class BusinessSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_overview: str = Field(max_length=2000)
    key_activities: list[SummaryText] = Field(max_length=10)
    customer_updates: list[SummaryText] = Field(max_length=10)
    task_updates: list[SummaryText] = Field(max_length=10)
    important_updates: list[SummaryText] = Field(max_length=10)
    operational_insights: list[SummaryText] = Field(max_length=10)
    data_limitations: list[SummaryText] = Field(max_length=10)

    @field_validator("business_overview")
    @classmethod
    def validate_overview(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("business_overview must not be blank.")
        return value

    @field_validator(
        "key_activities",
        "customer_updates",
        "task_updates",
        "important_updates",
        "operational_insights",
        "data_limitations"
    )
    @classmethod
    def normalize_summary_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class BusinessSummaryResponse(BaseModel):
    success: Literal[True] = True
    summary: BusinessSummary
