# Changelog

All notable changes to LOCO Detector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-19

### Added
- Initial release of LOCO Detector as a standalone program
- Scribble-based interactive segmentation with real-time feedback
- LOCO Circle Detection pipeline with ML models (Random Forest, ExtraTrees, XGBoost, CatBoost, LightGBM)
- Diameter measurement with 3 methodology versions (v1, v2, v3)
- Hierarchical 3-group UI navigation with backward compatibility
- Scale calibration system (px → nm/μm) with persistent JSON storage
- Statistical distribution histogram with mean, median, std, min, max, and CSV export
- Pipeline connection: LOCO Detector accepted circles → Diameter Research points
- Validation system for manual review of detected circles
- LOCO Lab experimental pipeline (proposals → filter → measure → evaluate)
- Dataset management: build, augment, and manage training datasets
- Training system with binary + multiclass classification
- Circle-NMS and spatial final filter for overlap removal
- Comprehensive test suite (50+ tests)
- Security documentation and localhost-only binding

### Changed
- Migrated from Scribble Research library context to standalone LOCO Detector program
- Reorganized flat 9-tab navigation into 3 hierarchical groups
- Archived obsolete documentation to `archived/` directory

### Fixed
- Various spatial overlap issues in circle detection
- Pipeline connection between LOCO Detector and Diameter Research modules
