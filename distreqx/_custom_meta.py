from dataclasses import dataclass

import equinox as eqx
import ihoop


try:

    class _StrictEqxMeta(ihoop.strict._StrictMeta, eqx._module._module._ModuleMeta):
        def __new__(mcs, name, bases, namespace, **kwargs):
            return super().__new__(mcs, name, bases, namespace, **kwargs)

    @dataclass(frozen=True)
    class StrictModule(eqx.Module, ihoop.Strict, metaclass=_StrictEqxMeta):  # type: ignore[reportRedeclaration]
        def __init_subclass__(cls, *, strict: bool | None = None, **kwargs):
            super().__init_subclass__(**kwargs)
except Exception:

    class StrictModule(eqx.Module):
        pass
