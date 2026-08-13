"""
Six-category grid mapping for the Me Classification Engine.

This module handles mapping classified concepts onto a structured grid
with six categories:
1. Office & Regalia
2. Craft & Technical Power
3. Statutory Law & Protocol
4a. Internal Attribute
4b. Enacted Practice
6. Judicial Sanction / Breach
"""

from dataclasses import dataclass, field
from typing import Optional
from .models import CategorySlot, CategoryCode


@dataclass
class GridMapper:
    """
    Maps classified concepts onto the six-category grid.
    """
    
    # Category definitions. Each category carries single-word `keywords`
    # (word-boundary matched) and multi-word `phrases` (substring matched,
    # since they're specific enough not to false-positive). The keyword
    # lists were originally narrow enough that well-sourced content could
    # legitimately qualify for a slot but get scored null just because it
    # used different vocabulary than the hardcoded list (e.g. "prophetic
    # decree of institutional collapse" didn't match Slot 6's original
    # "penalty/punishment/exile" list even though it plainly describes a
    # judicial breach consequence). Phrases close that gap.
    CATEGORIES: dict[CategoryCode, dict] = field(default_factory=lambda: {
        CategoryCode.OFFICE_REGALIA: {
            "name": "Office & Regalia",
            "description": "Governance and priestly roles, plus attached regalia",
            "keywords": [
                "office", "throne", "crown", "scepter", "staff",
                "priest", "priestess", "king", "queen", "lord", "lady",
                "ruler", "sovereign", "magistrate", "judge",
                "vestment", "robe", "garment", "regalia", "insignia",
                "ephod", "hereditary", "title", "authority"
            ],
            "phrases": [
                "high priest", "priestly office", "seat of judgment",
                "chair of judgment", "hereditary office",
            ]
        },
        CategoryCode.CRAFT_TECHNICAL: {
            "name": "Craft & Technical Power",
            "description": "Skilled trades and technical capacities",
            "keywords": [
                "craft", "skill", "trade", "art", "technique",
                "carpentry", "smithing", "weaving", "pottery",
                "scribe", "writing", "building", "constructing",
                "engineering", "architecture", "measurement",
                "logistics", "transport", "masonry", "slaughter"
            ],
            "phrases": [
                "technical craft", "sanctuary logistics", "animal slaughter",
                "sacrificial processing",
            ]
        },
        CategoryCode.STATUTORY_LAW: {
            "name": "Statutory Law & Protocol",
            "description": "Institutional procedure and codified law",
            "keywords": [
                "law", "statute", "code", "decree", "ordinance",
                "protocol", "procedure", "regulation", "prescription",
                "ritual", "ceremony", "liturgy", "calendar",
                "tariff", "tax", "tribute", "allocation", "custom"
            ],
            "phrases": [
                "codified regulation", "assembly protocol", "custom of the priests",
            ]
        },
        CategoryCode.INTERNAL_ATTRIBUTE: {
            "name": "Internal Attribute",
            "description": "Conditions someone carries, not directed outward",
            "keywords": [
                "joy", "awe", "kindness", "reverence", "silence",
                "respect", "honor", "glory", "majesty", "splendor",
                "radiance", "beauty", "perfection", "completeness",
                "integrity", "piety", "accountability", "virtue"
            ],
            "phrases": [
                "divine weight", "cultic accountability",
            ]
        },
        CategoryCode.ENACTED_PRACTICE: {
            "name": "Enacted Practice",
            "description": "Acts involving another party or outward action",
            "keywords": [
                "kissing", "embrace", "sexual", "triumph", "victory",
                "strife", "conflict", "war", "battle", "movement",
                "settledness", "travel", "journey", "migration",
                "deceit", "trickery", "magic", "sorcery",
                "pilgrimage", "divination", "oracle", "offering"
            ],
            "phrases": [
                "burnt offering", "casting lots", "shrine divination",
            ]
        },
        CategoryCode.JUDICIAL_SANCTION: {
            "name": "Judicial Sanction / Breach",
            "description": "Legal consequences and breach penalties",
            "keywords": [
                "penalty", "punishment", "sanction", "fine",
                "exile", "imprisonment", "death", "execution",
                "breach", "violation", "transgression", "sin",
                "curse", "oath", "collapse", "downfall", "annihilation"
            ],
            "phrases": [
                "divine retribution", "institutional collapse",
                "prophetic decree", "structural annihilation",
                "loss of divine presence", "system failure",
                "cosmic collapse",
            ]
        }
    })
    
    def map_to_grid(
        self,
        subject: str,
        context: str,
        period_terms: Optional[list[str]] = None
    ) -> list[CategorySlot]:
        """
        Map a concept to the six-category grid.
        
        Args:
            subject: The concept being mapped
            context: Additional context about the concept
            period_terms: Source-language terms (transliterated)
            
        Returns:
            List of CategorySlots with is_null=True for empty slots
        """
        grid: list[CategorySlot] = []
        context_lower = context.lower()
        subject_lower = subject.lower()
        
        for code, category in self.CATEGORIES.items():
            # Substring match for single keywords, same as original behavior
            # (not switched to word-boundary matching -- that broke plural/
            # compound forms like "office" no longer matching "offices").
            matching_keywords = [
                kw for kw in category["keywords"]
                if kw in context_lower or kw in subject_lower
            ]

            # Substring match for multi-word phrases -- this is the actual
            # fix: it catches genuine signal that the original single-word
            # keyword list missed (e.g. "institutional collapse" as a
            # Judicial Sanction signal even though neither "institutional"
            # nor "collapse" alone was in the original keyword list).
            matching_phrases = [
                ph for ph in category.get("phrases", [])
                if ph in context_lower or ph in subject_lower
            ]

            all_matches = matching_keywords + matching_phrases

            # Determine if slot should be null
            is_null = len(all_matches) == 0
            
            # Get period terms for this category
            terms_for_slot: list[str] = []
            if period_terms:
                # In a real implementation, we'd have a mapping from
                # period terms to categories. For now, use all terms
                # for non-null slots
                if not is_null:
                    terms_for_slot = period_terms
            
            # Generate gloss from matching keywords/phrases (phrases first,
            # since they're more specific/informative than single words)
            gloss = ""
            if not is_null:
                display_matches = matching_phrases[:2] + matching_keywords[:2]
                gloss = f"Contains: {', '.join(display_matches[:3])}"
            
            grid.append(CategorySlot(
                code=code,
                name=category["name"],
                period_terms=terms_for_slot,
                gloss=gloss,
                is_null=is_null
            ))
        
        return grid
    
    def get_grid_summary(self, grid: list[CategorySlot]) -> str:
        """
        Generate a summary of the grid.
        
        Args:
            grid: List of CategorySlots
            
        Returns:
            Formatted summary string
        """
        lines = ["Category Grid:"]
        lines.append("-" * 60)
        
        null_count = sum(1 for slot in grid if slot.is_null)
        filled_count = len(grid) - null_count
        
        for slot in grid:
            lines.append(slot.to_display_string())
        
        lines.append("-" * 60)
        lines.append(f"Filled: {filled_count}/{len(grid)} | Null: {null_count}/{len(grid)}")
        
        return "\n".join(lines)


def create_empty_grid() -> list[CategorySlot]:
    """
    Create an empty grid with all slots set to is_null=True.
    
    Returns:
        List of CategorySlots with is_null=True
    """
    mapper = GridMapper()
    return [
        CategorySlot(
            code=code,
            name=info["name"],
            is_null=True
        )
        for code, info in mapper.CATEGORIES.items()
    ]
