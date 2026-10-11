"""Legacy import facade for the model registry and notebook sessions."""
import sys
from fuzzydiff.models import registry as implementation
sys.modules[__name__] = implementation
