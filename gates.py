"""
Gate evaluators for the Me Classification Engine.

Each gate evaluates a specific criterion for classification:
1. Philological Grounding - Lexical root exists and scales to administrative usage
2. Positive Ontology - Substance-ontology vs non-dual/fluid ontology
3. Affirmative Structure - Positive, dogmatic, codified duties vs aporetic practice
"""

import re
from abc import ABC, abstractmethod
from typing import Optional
from .models import GateResult, GateName, ConfidenceTier


def word_boundary_match(term: str, text: str) -> bool:
    """
    Check if term appears as a whole word in text.
    
    Args:
        term: Term to search for
        text: Text to search in
        
    Returns:
        True if term appears as whole word
    """
    pattern = r'\b' + re.escape(term) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


class GateEvaluator(ABC):
    """
    Base class for gate evaluators.
    
    This class provides the interface for evaluating a candidate concept
    against a specific gate criterion.
    """
    
    @abstractmethod
    def evaluate(self, subject: str, context: str, sources: list[str]) -> GateResult:
        """
        Evaluate a subject against this gate.
        
        Args:
            subject: The concept being evaluated
            context: Additional context (etymology notes, source citations)
            sources: List of source citations provided
            
        Returns:
            GateResult with pass/fail and reasoning
        """
        pass


class PhilologicalGroundingGate(GateEvaluator):
    """
    Gate 1: Philological Grounding
    
    Does a Tier 1 or Tier 2 lexical root exist, connecting the term to a
    concrete physical/mechanical/bodily action, AND does that root scale
    toward administrative/legal usage?
    
    The root must trend toward engineering-to-statute:
    - plumb-line → cosmic order
    - stake-driving → pact
    - measuring → justice
    """
    
    # Known lexical roots with administrative scaling
    # These require EXACT match in subject or explicit mention in context
    KNOWN_ROOTS: dict[str, dict] = {
        # Egyptian
        "mꜥꜣ": {
            "tier": 1,
            "source": "Gardiner, Egyptian Grammar",
            "scaling": "plumb-line → cosmic order → justice",
            "pass": True
        },
        # Hebrew
        "yšp": {
            "tier": 1,
            "source": "BDB Hebrew Lexicon",
            "scaling": "judge → rule → establish order",
            "pass": True
        },
        "qwm": {
            "tier": 1,
            "source": "HALOT Hebrew Lexicon",
            "scaling": "stand up → establish → ordain",
            "pass": True
        },
        "hwh": {
            "tier": 1,
            "source": "BDB Hebrew Lexicon",
            "scaling": "to be, to exist → eternal self-existence → divine name",
            "pass": True
        },
        # Sumerian - requires explicit mention of Sumerian context
        "me": {
            "tier": 2,
            "source": "ePSD, ETCSL corpus attestations",
            "scaling": "workforce → administrative cohort → institutional authority",
            "pass": True,
            "requires_context": ["sumerian", "cuneiform", "parṣu"]
        },
        # Vedic
        "ṛ": {
            "tier": 1,
            "source": "Monier-Williams Sanskrit Dictionary",
            "scaling": "flow → cosmic order → ritual law",
            "pass": True
        },
        # Latin
        "pax": {
            "tier": 1,
            "source": "Lewis & Short Latin Dictionary",
            "scaling": "peace → treaty → divine accord",
            "pass": True
        },
    }
    
    # Recognized philological reference works. A cited root is only credited
    # as genuinely grounded if paired with one of these -- not just "any
    # non-empty sources list", which was the previous (exploitable) check.
    RECOGNIZED_LEXICONS: list[str] = [
        "bdb", "brown-driver-briggs", "halot", "köhler", "kohler", "baumgartner",
        "gardiner", "monier-williams", "lewis & short", "lewis and short",
        "liddell", "liddell-scott", "epsd", "etcsl", "chicago assyrian dictionary",
        "cad", "von soden", "tdot", "dch", "assyrian dictionary",
        "cleasby-vigfusson", "cleasby", "vigfusson", "vigfússon", "zoega",
    ]

    # Matches phrasing like: root 'z-b-h', root z-b-h, root "yšp", from root mꜥꜣ
    ROOT_CITATION_PATTERN = re.compile(
        r"root\s+['\"]?([^\s,.;()'\"]+)", re.IGNORECASE
    )

    # Common English words that can follow "root" without being the cited
    # root itself (e.g. "root for structural enclosure", "root of the term").
    # Without this filter, prose like "Semitic root for X" would misfire.
    ROOT_STOPWORDS: set[str] = {
        "for", "of", "in", "to", "the", "that", "meaning", "connecting",
        "is", "was", "from", "and", "with", "which", "this", "a", "an",
    }

    def _extract_cited_root(self, context: str) -> Optional[str]:
        """Pull a candidate root out of 'root X' / "root 'X'" phrasing.

        Scans all "root ..." occurrences and returns the first one that
        isn't just common English filler, so prose like "Semitic root for
        structural enclosure ... from root z-b-h" correctly finds z-b-h
        rather than stopping at "for".
        """
        for match in self.ROOT_CITATION_PATTERN.finditer(context):
            candidate = match.group(1).strip("'\".,;()")
            if candidate.lower() not in self.ROOT_STOPWORDS and candidate:
                return candidate
        return None

    def _find_recognized_lexicon(self, context_lower: str, sources: list[str]) -> Optional[str]:
        """Return the matched lexicon name if a recognized reference work is cited."""
        haystack = context_lower + " " + " ".join(s.lower() for s in sources)
        for lex in self.RECOGNIZED_LEXICONS:
            if lex in haystack:
                return lex
        return None

    # Patterns indicating administrative scaling (require whole word match)
    ADMINISTRATIVE_PATTERNS: list[str] = [
        "law", "statute", "decree", "order", "justice", "judgment",
        "office", "authority", "ritual", "prescription", "ordinance",
        "covenant", "treaty", "pact", "oath", "protocol",
        "worship", "sacrifice", "obligation", "duty", "command",
        "prohibition", "requirement", "mandate", "injunction"
    ]
    
    def evaluate(self, subject: str, context: str, sources: list[str]) -> GateResult:
        """
        Evaluate philological grounding.
        
        A pass requires:
        1. A Tier 1 or Tier 2 lexical root exists (with exact matching)
        2. The root connects to concrete physical/mechanical action
        3. The root scales toward administrative/legal usage
        """
        reasoning_parts: list[str] = []
        sources_found: list[str] = []
        confidence = ConfidenceTier.TIER_3  # Default to lowest
        
        # Check if we have explicit source citations
        has_explicit_sources = len(sources) > 0 and any(
            s.strip() for s in sources
        )
        
        # Check context for lexical roots (with exact matching)
        context_lower = context.lower()
        subject_lower = subject.lower()
        
        # Check for known roots (with stricter matching)
        root_found = False
        
        for root, info in self.KNOWN_ROOTS.items():
            # For roots with requires_context, check if context mentions the language
            if "requires_context" in info:
                has_required_context = any(
                    ctx in context_lower or ctx in subject_lower
                    for ctx in info["requires_context"]
                )
                if not has_required_context:
                    continue
            
            # Check for exact match in subject or explicit mention in context
            if (word_boundary_match(root, subject_lower) or 
                f"root {root}" in context_lower or
                f"root '{root}'" in context_lower or
                f"root \"{root}\"" in context_lower):
                root_found = True
                confidence = ConfidenceTier(info["tier"])
                sources_found.append(info["source"])
                reasoning_parts.append(
                    f"Lexical root '{root}' attested in {info['source']} "
                    f"(Tier {info['tier']}). Semantic scaling: {info['scaling']}."
                )
                break
        
        # Check for administrative patterns in context (require whole word)
        has_administrative = any(
            word_boundary_match(pattern, context_lower)
            for pattern in self.ADMINISTRATIVE_PATTERNS
        )
        
        # Check for physical/mechanical grounding
        physical_patterns = [
            "measure", "build", "construct", "plow", "weave", "forge",
            "write", "inscribe", "carve", "mold", "shape", "form",
            "drive", "strike", "plant", "sow", "harvest"
        ]
        has_physical = any(
            word_boundary_match(pattern, context_lower)
            for pattern in physical_patterns
        )
        
        # If no whitelisted root matched, try general root recognition:
        # a root explicitly cited via "root X" / "root 'X'" phrasing AND
        # paired with a recognized philological reference work. This lets
        # genuinely well-grounded roots outside the small hardcoded whitelist
        # (e.g. Hebrew z-b-h) pass on their own merits, without falling back
        # to "any non-empty sources list", which could be satisfied by a
        # fabricated citation with no real lexical verification at all.
        if not root_found:
            cited_root = self._extract_cited_root(context)
            matched_lexicon = self._find_recognized_lexicon(context_lower, sources)
            if cited_root and matched_lexicon and (has_administrative or has_physical):
                root_found = True
                confidence = ConfidenceTier.TIER_2
                sources_found.append(matched_lexicon)
                reasoning_parts.append(
                    f"Lexical root '{cited_root}' cited against recognized "
                    f"reference work ({matched_lexicon}). "
                    f"{'Administrative' if has_administrative else 'Physical/mechanical'} "
                    f"scaling attested in context."
                )

        # Determine pass/fail
        if root_found and has_administrative:
            passed = True
            reasoning_parts.append(
                "Root scales toward administrative/legal usage as required."
            )
        elif root_found and has_physical:
            # Physical grounding present but administrative scaling unclear
            passed = True
            reasoning_parts.append(
                "Physical/mechanical grounding present. Administrative scaling "
                "inferred from context."
            )
            confidence = ConfidenceTier.TIER_2
        else:
            # Determine if this is INDETERMINATE or FAIL
            if not has_explicit_sources and not root_found:
                # Insufficient information - return INDETERMINATE
                reasoning_parts.append(
                    "Insufficient source material provided to make determination."
                )
                return GateResult(
                    gate_name=GateName.PHILOLOGICAL_GROUNDING,
                    passed=False,
                    reasoning=" ".join(reasoning_parts),
                    confidence_tier=ConfidenceTier.TIER_3,
                    sources=sources_found,
                    indeterminate=True
                )
            
            passed = False
            if not root_found:
                reasoning_parts.append(
                    "No attested Tier 1 or Tier 2 lexical root found."
                )
            if not has_administrative and not has_physical:
                reasoning_parts.append(
                    "No evidence of physical/mechanical grounding or "
                    "administrative scaling."
                )
            if not has_explicit_sources:
                reasoning_parts.append(
                    "No source citations provided to support etymology claims."
                )
        
        # Add any provided sources
        sources_found.extend([s for s in sources if s.strip()])
        
        return GateResult(
            gate_name=GateName.PHILOLOGICAL_GROUNDING,
            passed=passed,
            reasoning=" ".join(reasoning_parts),
            confidence_tier=confidence,
            sources=list(dict.fromkeys(sources_found))  # Deduplicate while preserving order
        )


class PositiveOntologyGate(GateEvaluator):
    """
    Gate 2: Positive Ontology
    
    Does the system affirm a substance-ontology — discrete, bounded entities
    (gods, offices, civic roles, fixed legal relations) — rather than a
    relational, non-dual, or fluid ontology?
    """
    
    # Patterns indicating positive/substance ontology (require whole word)
    POSITIVE_ONTOLOGY_PATTERNS: list[str] = [
        # Discrete entities
        "god", "goddess", "deity", "lord", "lady", "king", "queen",
        "priest", "priestess", "office", "throne", "crown", "scepter",
        # Fixed relations
        "covenant", "treaty", "pact", "oath", "statute",
        "decree", "ordinance", "prescription", "protocol",
        # Bounded roles
        "judge", "ruler", "sovereign", "magistrate", "overseer",
        "administrator", "steward", "officer"
    ]
    
    # Patterns indicating non-dual/fluid ontology (require whole word)
    NON_DUAL_PATTERNS: list[str] = [
        "non-dual", "nondual", "advaita", "emptiness", "sunyata",
        "flux", "change", "becoming", "impermanence",
        "unmanifest", "formless", "boundless", "infinite",
        "mystery", "ineffable", "unknowable",
        "dissolution", "unity", "oneness", "totality",
        "dreaming", "dreamtime", "songline"
    ]
    
    def evaluate(self, subject: str, context: str, sources: list[str]) -> GateResult:
        """
        Evaluate positive ontology.
        
        A pass requires:
        1. System affirms discrete, bounded entities
        2. System affirms fixed legal relations
        3. System does NOT primarily function as non-dual/fluid ontology
        """
        reasoning_parts: list[str] = []
        
        context_lower = context.lower()
        subject_lower = subject.lower()
        
        # Count positive ontology indicators (require whole word)
        positive_count = sum(
            1 for pattern in self.POSITIVE_ONTOLOGY_PATTERNS
            if word_boundary_match(pattern, context_lower) or 
               word_boundary_match(pattern, subject_lower)
        )
        
        # Count non-dual indicators (require whole word)
        non_dual_count = sum(
            1 for pattern in self.NON_DUAL_PATTERNS
            if word_boundary_match(pattern, context_lower) or 
               word_boundary_match(pattern, subject_lower)
        )
        
        # Check for explicit ontology claims (with negative detection)
        has_substance_ontology = any(
            phrase in context_lower
            for phrase in [
                "substance ontology", "discrete entities", "bounded",
                "fixed relations", "categorical"
            ]
        ) and not any(
            phrase in context_lower
            for phrase in ["not discrete", "not bounded", "not categorical"]
        )
        
        has_fluid_ontology = any(
            phrase in context_lower
            for phrase in [
                "relational ontology", "process ontology", "fluid",
                "non-substantial", "non-categorical"
            ]
        ) or any(
            phrase in context_lower
            for phrase in [
                "not bounded", "not discrete", "no bounded",
                "no discrete", "fluid ontology"
            ]
        )
        
        # Check for explicit negation of substance ontology
        has_negated_substance = any(
            phrase in context_lower
            for phrase in [
                "not bounded entities", "not discrete entities",
                "no bounded entities", "no discrete entities",
                "not substance ontology"
            ]
        )
        
        # Determine pass/fail
        if has_negated_substance or has_fluid_ontology:
            passed = False
            reasoning_parts.append(
                "Explicit fluid/relational ontology claims found in source material."
            )
            if has_negated_substance:
                reasoning_parts.append(
                    "Substance-ontology explicitly negated."
                )
        elif has_substance_ontology and not has_fluid_ontology:
            passed = True
            reasoning_parts.append(
                "Explicit substance-ontology affirmed in source material."
            )
        elif positive_count > non_dual_count and positive_count >= 2:
            passed = True
            reasoning_parts.append(
                f"Discrete entities and fixed relations present "
                f"({positive_count} indicators vs {non_dual_count} non-dual indicators)."
            )
        elif non_dual_count > positive_count:
            passed = False
            reasoning_parts.append(
                f"Non-dual/fluid ontology indicators dominate "
                f"({non_dual_count} vs {positive_count} positive indicators)."
            )
            reasoning_parts.append(
                "System appears to affirm relational/process ontology "
                "rather than substance-ontology."
            )
        else:
            # Insufficient information
            passed = False
            reasoning_parts.append(
                "Insufficient information to determine ontology type. "
                "No clear indicators of substance-ontology found."
            )
        
        return GateResult(
            gate_name=GateName.POSITIVE_ONTOLOGY,
            passed=passed,
            reasoning=" ".join(reasoning_parts),
            confidence_tier=ConfidenceTier.TIER_2 if passed else ConfidenceTier.TIER_3,
            sources=[s for s in sources if s.strip()]
        )


class AffirmativeStructureGate(GateEvaluator):
    """
    Gate 3: Affirmative Structure
    
    Does the system prescribe positive, dogmatic, codified duties/law
    (statutes, calendars, tariffs) rather than functioning as an aporetic,
    therapeutic, or purely diplomatic practice?
    """
    
    # Patterns indicating affirmative/positive structure (require whole word)
    AFFIRMATIVE_PATTERNS: list[str] = [
        # Codified law
        "law", "statute", "code", "codex", "decree", "ordinance",
        "regulation", "prescription", "protocol", "procedure",
        # Calendar/ritual
        "calendar", "festival", "ritual", "sacrifice", "offering",
        "hymn", "prayer", "liturgy", "worship", "ceremony",
        # Tariffs/economics
        "tariff", "tax", "toll", "tribute", "allocation",
        "ration", "distribution", "redistribution",
        # Dogmatic claims
        "command", "prohibition", "requirement", "obligation",
        "duty", "responsibility", "mandate", "injunction"
    ]
    
    # Patterns indicating aporetic/therapeutic practice (require whole word)
    APORNETIC_PATTERNS: list[str] = [
        "aporia", "paradox", "contradiction", "mystery", "enigma",
        "koan", "meditation", "contemplation", "reflection",
        "therapeutic", "healing", "cure", "remedy", "treatment",
        "diplomatic", "negotiation", "compromise", "reconciliation",
        "inquiry", "questioning", "doubt", "uncertainty",
        "silence", "ineffable", "unknowable"
    ]
    
    def evaluate(self, subject: str, context: str, sources: list[str]) -> GateResult:
        """
        Evaluate affirmative structure.
        
        A pass requires:
        1. System prescribes positive, dogmatic, codified duties
        2. System functions as statute/calendar/tariff
        3. System does NOT primarily function as aporetic/therapeutic practice
        """
        reasoning_parts: list[str] = []
        
        context_lower = context.lower()
        subject_lower = subject.lower()
        
        # Count affirmative structure indicators (require whole word)
        affirmative_count = sum(
            1 for pattern in self.AFFIRMATIVE_PATTERNS
            if word_boundary_match(pattern, context_lower) or 
               word_boundary_match(pattern, subject_lower)
        )
        
        # Count aporetic indicators (require whole word)
        aporetic_count = sum(
            1 for pattern in self.APORNETIC_PATTERNS
            if word_boundary_match(pattern, context_lower) or 
               word_boundary_match(pattern, subject_lower)
        )
        
        # Check for explicit structure claims (with negative detection)
        has_codified_law = any(
            phrase in context_lower
            for phrase in [
                "codified", "written law", "legal code",
                "prescriptive", "regulatory", "mandatory"
            ]
        ) and not any(
            phrase in context_lower
            for phrase in ["no codified", "not codified", "without codified"]
        )
        
        has_aporetic_practice = any(
            phrase in context_lower
            for phrase in [
                "aporetic", "therapeutic", "contemplative", "meditative",
                "mystical practice", "silent", "ineffable"
            ]
        ) or any(
            phrase in context_lower
            for phrase in [
                "not prescriptive", "no codified", "no statutory",
                "not statutory", "not legal"
            ]
        )
        
        # Check for explicit negation of codified structure
        has_negated_structure = any(
            phrase in context_lower
            for phrase in [
                "no codified statutory", "not codified statutory",
                "no statutory structure", "not statutory structure",
                "no prescriptive", "not prescriptive"
            ]
        )
        
        # Determine pass/fail
        if has_negated_structure or has_aporetic_practice:
            passed = False
            reasoning_parts.append(
                "Explicit aporetic/therapeutic practice claims found."
            )
            if has_negated_structure:
                reasoning_parts.append(
                    "Codified/prescriptive structure explicitly negated."
                )
        elif has_codified_law and not has_aporetic_practice:
            passed = True
            reasoning_parts.append(
                "Explicit codified/prescriptive structure affirmed."
            )
        elif affirmative_count > aporetic_count and affirmative_count >= 2:
            passed = True
            reasoning_parts.append(
                f"Positive, codified duties present "
                f"({affirmative_count} indicators vs {aporetic_count} aporetic indicators)."
            )
        elif aporetic_count > affirmative_count:
            passed = False
            reasoning_parts.append(
                f"Aporietic/therapeutic indicators dominate "
                f"({aporetic_count} vs {affirmative_count} affirmative indicators)."
            )
            reasoning_parts.append(
                "System appears to function as contemplative/therapeutic "
                "practice rather than prescriptive law."
            )
        else:
            # Insufficient information
            passed = False
            reasoning_parts.append(
                "Insufficient information to determine structural type. "
                "No clear indicators of affirmative/prescriptive structure found."
            )
        
        return GateResult(
            gate_name=GateName.AFFIRMATIVE_STRUCTURE,
            passed=passed,
            reasoning=" ".join(reasoning_parts),
            confidence_tier=ConfidenceTier.TIER_2 if passed else ConfidenceTier.TIER_3,
            sources=[s for s in sources if s.strip()]
        )


def get_all_gate_evaluators() -> list[GateEvaluator]:
    """
    Get all gate evaluators in order.
    
    Returns:
        List of GateEvaluator instances in evaluation order
    """
    return [
        PhilologicalGroundingGate(),
        PositiveOntologyGate(),
        AffirmativeStructureGate()
    ]
