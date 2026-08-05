from pydantic import BaseModel, Field


class Decision(BaseModel):
    """A concrete technical decision made in the project."""
    title: str = Field(description="Short title of the decision")
    summary: str = Field(description="1-2 sentence description")
    tags: list[str] = Field(default_factory=list, description="Topic tags like 'db', 'ui', 'auth'")


class Rule(BaseModel):
    """A guideline or convention developers should follow."""
    rule: str = Field(description="The rule itself, in one clear sentence")
    scope: str = Field(description="Where it applies: 'ui', 'api', 'styling', 'general', etc.")
    notes: str = Field(default="", description="Optional caveats or exceptions")


class Warning(BaseModel):
    """A sensitive area, risky operation, or thing to avoid."""
    area: str = Field(description="What part of the system: 'auth', 'cart', 'routing', etc.")
    message: str = Field(description="The warning itself")
    severity: str = Field(description="One of: 'low', 'medium', 'high'")


class ExtractedItems(BaseModel):
    """All items extracted from a single documentation file."""
    decisions: list[Decision] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
