def unknown_component_error(component_type: str, name: str, registry: dict):
    available = ", ".join(sorted(str(k) for k in registry.keys()))
    return ValueError(
        f"Unknown {component_type} '{name}'. "
        f"Available options: [{available}]"
    )