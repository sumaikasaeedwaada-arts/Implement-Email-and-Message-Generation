class ChatServiceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.public_message = message


class ChatUnavailableError(ChatServiceError):
    pass


class InvalidChatMessageError(ChatServiceError):
    pass
