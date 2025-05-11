# Tree Risk Pro Changelog

> This is the master changelog for the entire Tree Risk Pro project.
> For component-specific changelogs, see:
> - [Dashboard Changelog](/tree_risk_pro/dashboard/CHANGELOG.md)

## [v0.2.1] - 2025-05-11

### Added
- S2 geospatial indexing with Zarr store integration
- Validation reports linking to area reports via S2 cells
- New API endpoints for S2 cell-based report management
- Enhanced Object Report view with linked validation reports
- Improved ML overlay with persistent opacity settings
- Click-through functionality for ML overlay to improve marker interaction
- Enhanced error handling for DOM operations
- Header collapse state detection for better UI coordination
- Event-based coordination between UI components

### Changed
- Updated version numbers across all components to v0.2.1
- Improved ML overlay creation process with gradient background
- Enhanced sidebar management with custom events
- Enhanced Reports Overview with S2 region indicators
- Consolidated documentation for better consistency

### Fixed
- Components/Detection sidebar functionality with proper event handling
- ML overlay opacity control with smoother transitions
- OBJECT DETECTION badge visibility with correct z-index (2000)
- Analytics panel closing when opening Components panel
- Sidebar panel management for better user experience
- Map container resizing during sidebar transitions
- DOM element cleanup to prevent ghost elements
- Performance issues when switching between detection modes

## [v0.2.0] - 2025-04-27

### Added
- Separator lines in sidebar views
- Full backend code documentation
- Better error logging
- Updated GCP deployment script
- Expanded deployment documentation

### Changed
- "Save to Database" → "Save for Review"
- Settings button moved to header
- Standardized header spacing (space-x-5)
- Improved GeminiService documentation
- Updated docs for v0.2.0

### Fixed
- 3D map state preservation between views
- Map resizing when toggling sidebars
- Header spacing consistency
- Navigation event handling
- Gemini API integration issues
- Removed debug console.log statements

## [v0.1.0] - 2025-02-15

### Added
- Initial release with core functionality
- 2D/3D map visualization
- LiDAR & Gemini AI tree detection
- Risk assessment workflow
- Report generation
- API security

### Changed
- UI built with shadcn/UI and Tailwind
- Backend on Flask with REST API
- Initial GCP deployment setup

### Fixed
- Initial stability improvements