"""Forward deadlines to the pinned SDK's actual transport calls.

langchain-google-genai 2.0.4 exposes a timeout field but does not forward it
from chat _generate/_stream. Keep the normal LangChain interface while fixing
that boundary, including query rewriting and diagnostic calls.
"""
from langchain_google_genai import ChatGoogleGenerativeAI


class BoundedGeminiChat(ChatGoogleGenerativeAI):
    def _request_options(self, kwargs):
        return {"timeout": self.timeout, "retry": None, **kwargs}

    def _generate(self, *args, **kwargs):
        return super()._generate(*args, **self._request_options(kwargs))

    def _stream(self, *args, **kwargs):
        yield from super()._stream(*args, **self._request_options(kwargs))

    async def _agenerate(self, *args, **kwargs):
        return await super()._agenerate(*args, **self._request_options(kwargs))

    async def _astream(self, *args, **kwargs):
        async for chunk in super()._astream(*args, **self._request_options(kwargs)):
            yield chunk
