import logging
import os
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger("jira_client")


class JiraClient:
    """
    Jira Cloud client with safe demo fallback.

    Modes:
        - "mock": always returns mock data
        - "real": only real Jira data, errors — no fallback
        - "auto": tries to get real, with any problem — fallback on the mock
    """

    def __init__(self):
        self.url = os.getenv("JIRA_URL", "").rstrip("/")
        self.email = os.getenv("JIRA_EMAIL")
        self.token = os.getenv("JIRA_API_TOKEN")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "PROJ")
        self.mode = os.getenv("JIRA_MODE", "auto").lower()

        self.enabled = bool(self.url and self.email and self.token)

        # Mode validation
        if self.mode not in {"mock", "real", "auto"}:
            logger.warning(f"Invalid JIRA_MODE='{self.mode}', fallback to 'auto'")
            self.mode = "auto"

        # Forced mock if there are no credentials
        if not self.enabled and self.mode != "mock":
            logger.warning("Jira credentials missing → forcing MOCK mode")
            self.mode = "mock"

        if self.mode == "mock":
            logger.info("🧪 Jira client running in MOCK mode")
        else:
            logger.info(
                "✅ Jira client initialized",
                extra={
                    "url": self.url,
                    "project": self.project_key,
                    "mode": self.mode,
                },
            )

    # =================================================================
    # Internal Helpers
    # =================================================================
    def _auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(self.email, self.token)

    def _mock_issues(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": f"{self.project_key}-101",
                "fields": {
                    "summary": "Project initialization and setup",
                    "status": {"name": "Done"},
                    "assignee": {"displayName": "Developer"},
                    "created": "2025-12-01T10:00:00.000+0000",
                },
            },
            {
                "key": f"{self.project_key}-102",
                "fields": {
                    "summary": "Implement multi-agent orchestration",
                    "status": {"name": "In Progress"},
                    "assignee": {"displayName": "Developer"},
                    "created": "2025-12-15T14:30:00.000+0000",
                },
            },
            {
                "key": f"{self.project_key}-103",
                "fields": {
                    "summary": "Add health monitoring and metrics",
                    "status": {"name": "To Do"},
                    "assignee": None,
                    "created": "2025-12-20T09:00:00.000+0000",
                },
            },
        ]

    async def _safe_request(self, method: str, url: str, **kwargs) -> Optional[dict]:
        """Secure request with error handling"""
        try:
            logger.info(f"Making {method} request to {url}")
            if "params" in kwargs:
                logger.info(f"Request params: {kwargs['params']}")

            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.request(
                    method,
                    url,
                    auth=self._auth(),
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    **kwargs,
                )

                logger.info(f"Response status: {response.status_code}")

                response.raise_for_status()
                data = response.json()

                # Log response structure
                if isinstance(data, dict):
                    logger.info(f"Response keys: {list(data.keys())}")
                    if "issues" in data:
                        logger.info(f"Issues count in response: {len(data.get('issues', []))}")

                return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "Jira HTTP error",
                extra={
                    "status": e.response.status_code,
                    "text": e.response.text[:500],  # Increased the limit for more context
                    "url": url,
                },
            )
        except Exception as e:
            logger.error(
                "Jira request failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "url": url,
                }
            )
        return None

    # =================================================================
    # Public methods
    # =================================================================
    async def get_project_issues(self, max_results: int = 50) -> Dict[str, Any]:
        """
        Get the project tasks.
        Returns a unified dict: {"issues": [...], "mode": "..."}
        """
        if self.mode == "mock":
            logger.info("Returning mock issues (mock mode)")
            return {
                "issues": self._mock_issues(),
                "mode": "mock",
            }

        # Construct JQL query
        jql = f"project = {self.project_key}"
        logger.info(f"Executing JQL query: {jql}")

        data = await self._safe_request(
            "GET",
            f"{self.url}/rest/api/3/search",
            params={
                "jql": jql,
                "maxResults": max_results,
                "fields": "summary,status,assignee,created",
            },
        )

        # Check if request failed
        if data is None:
            logger.error("Failed to get data from Jira API")

            if self.mode == "real":
                logger.info("Real mode: returning empty list (no fallback)")
                return {
                    "issues": [],
                    "mode": "real",
                }

            # auto mode → fallback on mock
            logger.info("Auto mode: falling back to mock issues due to API failure")
            return {
                "issues": self._mock_issues(),
                "mode": "mock_fallback",
            }

        # Extract issues from response
        issues = data.get("issues", [])
        total = data.get("total", 0)

        logger.info(f"Jira API returned {len(issues)} issues (total: {total})")

        # No data or an empty task list
        if not issues:
            logger.warning(f"No issues found for project {self.project_key}")

            if self.mode == "real":
                logger.info("Real mode: returning empty list (no fallback)")
                return {
                    "issues": [],
                    "mode": "real",
                }

            # auto mode → fallback on mock
            logger.info("Auto mode: falling back to mock issues (empty result)")
            return {
                "issues": self._mock_issues(),
                "mode": "mock_fallback",
            }

        logger.info(f"Successfully fetched {len(issues)} real issues")
        return {
            "issues": issues,
            "mode": "real",
        }

    async def create_task(
            self,
            summary: str,
            description: str = "",
            issue_type: str = "Task",
    ) -> Dict[str, Any]:
        """Create a task in Jira"""
        if self.mode == "mock":
            fake_key = f"{self.project_key}-{hash(summary) % 10000:04d}"
            return {
                "status": "created",
                "issue_key": fake_key,
                "url": f"https://mock-jira.atlassian.net/browse/{fake_key}",
                "mode": "mock",
            }

        payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description or summary}],
                        }
                    ],
                },
                "issuetype": {"name": issue_type},
            }
        }

        data = await self._safe_request("POST", f"{self.url}/rest/api/3/issue", json=payload)

        if not data or "key" not in data:
            if self.mode == "auto":
                logger.warning("Task creation failed → returning mock result")
                fake_key = f"{self.project_key}-{hash(summary) % 10000:04d}"
                return {
                    "status": "created",
                    "issue_key": fake_key,
                    "mode": "mock_fallback",
                }
            return {
                "status": "error",
                "error": "Failed to create issue",
                "mode": "real",
            }

        issue_key = data["key"]
        return {
            "status": "created",
            "issue_key": issue_key,
            "url": f"{self.url}/browse/{issue_key}",
            "mode": "real",
        }
