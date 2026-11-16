"""Extract bounded retail sales requests with Bedrock and local safeguards."""

import re

from botocore.exceptions import BotoCoreError, ClientError

from agent.planning_contract import PlanningContract


STORE_PATTERN = re.compile(
    r"\bstores?\s*(?:number\s*|#\s*)?"
    r"(\d+(?:\s*(?:,|and|&)\s*(?:store\s*)?\d+)*)\b",
    flags=re.IGNORECASE,
)

PERIOD_PATTERNS = (
    (r"\bnext two weeks\b", "next two weeks"),
    (r"\bnext 2 weeks\b", "next two weeks"),
    (r"\bnext 14 days\b", "next two weeks"),
    (r"\bnext seven days\b", "next 7 days"),
    (r"\bnext 7 days\b", "next 7 days"),
    (r"\bnext week(?:'s)?\b", "next week"),
    (r"\btomorrow\b", "tomorrow"),
)

UNSUPPORTED_OPERATIONS = re.compile(
    r"\b(?:highest|busiest|top|most|least|rank|ranking|"
    r"compare|comparison|deliver|delivery|deliveries|"
    r"replenish|replenishment|inventory|stock|"
    r"discount|promotion|order|orders)\b",
    flags=re.IGNORECASE,
)


class SalesRequestInterpreter:
    """Extract structured intent and verify explicit question constraints."""

    def __init__(
        self, bedrock_client, model_id, calendar, known_families
    ):
        self.bedrock = bedrock_client
        self.model_id = model_id
        self.calendar = calendar
        self.known_families = tuple(
            sorted({
                str(name).upper()
                for name in known_families
            })
        )

    def _literal_anchors(self, question):
        """Independent checks for the currently supported request class."""

        if UNSUPPORTED_OPERATIONS.search(question):
            return None, (
                "This request needs a different business operation."
            )

        if re.search(
            r"\b(?:or|either|if|unless)\b",
            question,
            flags=re.IGNORECASE,
        ):
            return None, (
                "Alternative or conditional requests are unsupported."
            )

        stores = set()

        for match in STORE_PATTERN.finditer(question):
            stores.update(
                int(value)
                for value in re.findall(r"\d+", match[1])
            )

        families = set()

        for family in self.known_families:
            pattern = (
                r"(?<!\w)"
                + re.escape(family).replace(r"\ ", r"\s+")
                + r"(?!\w)"
            )

            if re.search(pattern, question, flags=re.IGNORECASE):
                families.add(family)

        if not stores or not families:
            return None, (
                "Store(s) and product family must be named explicitly."
            )

        periods = [
            label
            for pattern, label in PERIOD_PATTERNS
            if re.search(
                pattern, question, flags=re.IGNORECASE
            )
        ]

        dates = re.findall(
            r"\b\d{4}-\d{2}-\d{2}\b",
            question,
        )

        if dates:
            if periods or len(dates) not in (1, 2):
                return None, "Ambiguous date selection."

            expression = "/".join(dates)

        elif len(periods) == 1:
            expression = periods[0]

        else:
            return None, (
                "Missing or ambiguous planning period."
            )

        try:
            period = self.calendar.resolve(expression)
        except (ValueError, TypeError):
            return None, (
                "Dates outside the available sales period."
            )

        return {
            "stores": stores,
            "families": families,
            "period": period,
            "readiness": bool(
                re.search(
                    r"\b(?:ready|available)\b",
                    question,
                    flags=re.IGNORECASE,
                )
            ),
        }, None

    def extract(self, question):
        """Return a validated contract or an explicit failure status."""

        if not isinstance(question, str) or not question.strip():
            return {
                "status": "needs_clarification",
                "reason": "Empty question.",
            }

        anchors, issue = self._literal_anchors(question)

        if anchors is None:
            return {
                "status": "unsupported",
                "reason": issue,
            }

        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "sales_total",
                        "unsupported",
                        "needs_clarification",
                    ],
                },
                "stores": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "families": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "period": {"type": "string"},
                "requires_readiness": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "intent",
                "stores",
                "families",
                "period",
                "requires_readiness",
                "reason",
            ],
        }

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{
                    "text": (
                        "Extract a retail planning request. "
                        "Use the tool to report its meaning. "
                        "Only select sales_total for a request asking "
                        "for expected sales quantities for explicitly "
                        "named stores and product families during an "
                        "explicitly named period. "
                        f"Known product families: {', '.join(self.known_families)}. "
                        "Each listed family is a complete category. "
                        "For example, DAIRY needs no further subcategory. "
                        "'Next week' and 'next two weeks' are valid "
                        "explicit planning periods resolved by the "
                        "application calendar. Do not request exact "
                        "calendar dates for these expressions. "
                        "Do not infer missing stores, dates or families. "
                        "If unclear, use needs_clarification. "
                        "If ranking, inventory, promotion or delivery "
                        "planning is required, use unsupported."
                    )
                }],
                messages=[{
                    "role": "user",
                    "content": [{"text": question}],
                }],
                toolConfig={
                    "tools": [{
                        "toolSpec": {
                            "name": "submit_planning_contract",
                            "description": (
                                "Return the interpreted planning request."
                            ),
                            "inputSchema": {"json": schema},
                        }
                    }],
                    "toolChoice": {
                        "tool": {
                            "name": "submit_planning_contract"
                        }
                    },
                },
                inferenceConfig={
                    "maxTokens": 700,
                    "temperature": 0.0,
                },
            )

        except (ClientError, BotoCoreError) as exc:
            return {
                "status": "model_error",
                "reason": str(exc),
            }

        blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )

        uses = [
            block["toolUse"]
            for block in blocks
            if "toolUse" in block
        ]

        if (
            response.get("stopReason") != "tool_use"
            or len(uses) != 1
            or uses[0].get("name") != "submit_planning_contract"
        ):
            return {
                "status": "invalid_extraction",
                "reason": "No unique structured contract.",
            }

        raw = uses[0].get("input")

        if not isinstance(raw, dict):
            return {
                "status": "invalid_extraction",
                "reason": "Invalid contract payload.",
            }

        model_intent = raw.get("intent")

        # Recover narrowly defined false clarification decisions.
        # Scope equality is checked below before any approval.
        explicit_sales_wording = bool(
            re.match(r"^\s*how\s+much\b", question, re.I)
        )

        locally_reconciled = (
            model_intent == "needs_clarification"
            and explicit_sales_wording
        )

        if model_intent != "sales_total" and not locally_reconciled:
            return {
                "status": "unsupported",
                "reason": str(
                    raw.get("reason", "Unsupported operation.")
                ),
                "raw": raw,
                "model_intent": model_intent,
            }

        try:
            if (
                not isinstance(raw.get("stores"), list)
                or not isinstance(raw.get("families"), list)
            ):
                raise ValueError(
                    "Wrong types for stores or families."
                )

            contract = PlanningContract.create(
                self.calendar,
                raw["period"],
                stores=raw["stores"],
                families=raw["families"],
            )

            if (
                contract.stores != frozenset(anchors["stores"])
                or contract.families != frozenset(anchors["families"])
                or contract.period != anchors["period"]
            ):
                raise ValueError(
                    "Extracted scope disagrees with explicit "
                    "question constraints."
                )

        except (KeyError, TypeError, ValueError) as exc:
            return {
                "status": "invalid_extraction",
                "reason": str(exc),
                "raw": raw,
            }

        return {
            "status": "supported",
            "contract": contract,
            "requires_readiness": anchors["readiness"],
            "raw": raw,
            "model_intent": model_intent,
            "clarification_overridden": locally_reconciled,
        }
