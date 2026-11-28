"""12-language localization overlay for menu polish & Depth Map export cards."""
from __future__ import annotations

UI_POLISH_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "menu_community_dev": "Community (In development)",
        "depth_export_maps_group": "Depth Map & Layer Exports",
        "depth_export_mesh_group": "3D Mesh Export",
    },
    "zh": {
        "menu_community_dev": "社区（开发中）",
        "depth_export_maps_group": "深度图与数据导出",
        "depth_export_mesh_group": "3D 网格导出",
    },
    "ja": {
        "menu_community_dev": "コミュニティ（開発中）",
        "depth_export_maps_group": "深度マップ・データ書き出し",
        "depth_export_mesh_group": "3D メッシュ書き出し",
    },
    "ko": {
        "menu_community_dev": "커뮤니티 (개발 중)",
        "depth_export_maps_group": "깊이 맵 및 데이터 내보내기",
        "depth_export_mesh_group": "3D 메시 내보내기",
    },
    "de": {
        "menu_community_dev": "Community (in Entwicklung)",
        "depth_export_maps_group": "Tiefenkarten- & Datenexport",
        "depth_export_mesh_group": "3D-Mesh-Export",
    },
    "fr": {
        "menu_community_dev": "Communauté (en développement)",
        "depth_export_maps_group": "Export cartes de profondeur et données",
        "depth_export_mesh_group": "Export maillage 3D",
    },
    "es": {
        "menu_community_dev": "Comunidad (en desarrollo)",
        "depth_export_maps_group": "Exportación de mapas y datos",
        "depth_export_mesh_group": "Exportación de malla 3D",
    },
    "it": {
        "menu_community_dev": "Community (in sviluppo)",
        "depth_export_maps_group": "Esportazione mappe e dati",
        "depth_export_mesh_group": "Esportazione mesh 3D",
    },
    "nl": {
        "menu_community_dev": "Community (in ontwikkeling)",
        "depth_export_maps_group": "Dieptekaart- en gegevensexport",
        "depth_export_mesh_group": "3D-meshexport",
    },
    "pl": {
        "menu_community_dev": "Społeczność (w trakcie tworzenia)",
        "depth_export_maps_group": "Eksport map głębi i danych",
        "depth_export_mesh_group": "Eksport siatki 3D",
    },
    "pt": {
        "menu_community_dev": "Comunidade (em desenvolvimento)",
        "depth_export_maps_group": "Exportação de mapas e dados",
        "depth_export_mesh_group": "Exportação de malha 3D",
    },
    "ru": {
        "menu_community_dev": "Сообщество (в разработке)",
        "depth_export_maps_group": "Экспорт карт глубины и данных",
        "depth_export_mesh_group": "Экспорт 3D-сетки",
    },
}


def get_ui_polish_pack(lang_code: str) -> dict[str, str]:
    """Return localization entries for the UI polish updates."""
    return UI_POLISH_TRANSLATIONS.get(lang_code, UI_POLISH_TRANSLATIONS["en"])
