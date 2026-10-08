"""Stable, typed errors exposed by the AI Analyst API."""


class AnalystError(Exception):
    status_code = 400
    code = "ANALYST_ERROR"


class SessionNotFound(AnalystError):
    status_code, code = 404, "SESSION_NOT_FOUND"


class UnsupportedFile(AnalystError):
    status_code, code = 415, "UNSUPPORTED_FILE"


class FileTooLarge(AnalystError):
    status_code, code = 413, "FILE_TOO_LARGE"


class PipelineError(AnalystError):
    code = "PIPELINE_ERROR"


class UnsafeSQL(AnalystError):
    code = "UNSAFE_SQL"


class QueryError(AnalystError):
    code = "QUERY_ERROR"
