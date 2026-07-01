import jax

# Must be set before any JAX arrays are initialized. Living here (rather than in
# individual test files) guarantees it runs before any test module import.
jax.config.update("jax_enable_x64", True)
