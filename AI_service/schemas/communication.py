from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CommunicationType = Literal["email", "message"]
CommunicationPurpose = Literal[
    "customer_support",
    "follow_up",
    "announcement",
    "invitation",
    "custom"
]
CommunicationTone = Literal[
    "professional",
    "friendly",
    "formal",
    "polite",
    "persuasive",
    "empathetic"
]
CommunicationLength = Literal["short", "medium", "long"]


class CommunicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    communication_type: CommunicationType
    purpose: CommunicationPurpose = "customer_support"
    tone: CommunicationTone = "professional"
    length: CommunicationLength = "medium"
    language: str = Field(default="English", min_length=1, max_length=100)
    recipient_name: str | None = Field(default=None, max_length=120)
    recipient_context: str | None = Field(default=None, max_length=2000)
    customer_identifier: str | None = Field(default=None, max_length=200)
    custom_purpose: str | None = Field(default=None, max_length=300)
    additional_instructions: str | None = Field(default=None, max_length=2000)
    sender_name: str | None = Field(default=None, max_length=120)

    @field_validator(
        "language",
        "recipient_name",
        "recipient_context",
        "customer_identifier",
        "custom_purpose",
        "additional_instructions",
        "sender_name"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_custom_purpose(self):
        if self.purpose == "custom" and not self.custom_purpose:
            raise ValueError(
                "custom_purpose is required when purpose is custom."
            )
        if self.purpose != "custom" and self.custom_purpose:
            raise ValueError(
                "custom_purpose is only allowed when purpose is custom."
            )
        if not self.language:
            raise ValueError("language must not be blank.")
        return self


class EmailContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=8000)

    @field_validator("subject", "body")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Generated email fields must not be blank.")
        return normalized


class MessageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Generated message must not be blank.")
        return normalized


class CommunicationResponse(BaseModel):
    success: Literal[True] = True
    content: EmailContent | MessageContent
