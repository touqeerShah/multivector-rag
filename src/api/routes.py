from src.api.text_routes import build_text_router


# Compatibility shim: the split apps now use `text_routes` and `visual_routes`
# directly, but keep this module importable for older callers.
router = build_text_router(include_official_colbert=True)
