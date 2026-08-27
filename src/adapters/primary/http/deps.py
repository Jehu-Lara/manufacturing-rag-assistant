from __future__ import annotations

from typing import cast

from fastapi import Request

from src.adapters.primary.http.rate_limit import RateLimiter
from src.core.config import Settings
from src.domain.ports import VectorStorePort
from src.features.query.use_cases import QueryUseCase


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_query_use_case(request: Request) -> QueryUseCase:
    return cast(QueryUseCase, request.app.state.query_use_case)


def get_vector_store(request: Request) -> VectorStorePort:
    return cast(VectorStorePort, request.app.state.vector_store)


def get_rate_limiter(request: Request) -> RateLimiter:
    return cast(RateLimiter, request.app.state.rate_limiter)
