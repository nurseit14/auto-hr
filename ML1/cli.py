"""Shared CLI settings and matching path for already processed resume text."""
from ML1.config import (
    DEFAULT_ALPHA, DEFAULT_CONFIDENCE_TARGET, DEFAULT_TOP_K, MatchingConfig,
)
from ML1.matcher import JobMatcher


def add_matching_arguments(parser):
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='Skill weight in [0, 1]; semantic weight is 1-alpha.')
    parser.add_argument('--confidence-target', type=float, default=DEFAULT_CONFIDENCE_TARGET,
                        help='Positive vacancy-skill confidence target.')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Positive maximum number of results.')


def matching_config(parser, args):
    try:
        return MatchingConfig(alpha=args.alpha, confidence_target=args.confidence_target, top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))


def match_processed_resume(text, skills=None, config=None):
    config = config if config is not None else MatchingConfig()
    # Reject unusable input before reading the index or loading a model.
    if not isinstance(text, str) or not text.strip():
        raise ValueError('resume_text must be a non-empty string; no ranking was performed.')
    matcher = JobMatcher(alpha=config.alpha, confidence_target=config.confidence_target)
    return matcher.match(text, resume_skills=skills, top_k=config.top_k)
