"""Data-source connectors. All return the tidy schema in ``funkyforelle.schema``."""
from .base import Connector
from .camels_ch_chem import CamelsChChem
from .foen_hydro import FoenHydro
from .meteoswiss import MeteoSwiss

REGISTRY: dict[str, type[Connector]] = {
    FoenHydro.name: FoenHydro,
    MeteoSwiss.name: MeteoSwiss,
    CamelsChChem.name: CamelsChChem,
}

__all__ = ["Connector", "FoenHydro", "MeteoSwiss", "CamelsChChem", "REGISTRY"]
