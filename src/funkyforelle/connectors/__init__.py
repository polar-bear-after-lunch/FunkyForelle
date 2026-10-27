"""Data-source connectors. All return the tidy schema in ``funkyforelle.schema``."""
from .base import Connector
from .camels_ch_chem import CamelsChChem
from .foen_hydro import FoenHydro
from .meteoswiss import MeteoSwiss
from .openmeteo_forecast import OpenMeteoForecast

REGISTRY: dict[str, type[Connector]] = {
    FoenHydro.name: FoenHydro,
    MeteoSwiss.name: MeteoSwiss,
    CamelsChChem.name: CamelsChChem,
    OpenMeteoForecast.name: OpenMeteoForecast,
}

__all__ = ["Connector", "FoenHydro", "MeteoSwiss", "CamelsChChem", "OpenMeteoForecast", "REGISTRY"]
