# Changelog

Changes to the Tree Risk Pro Dashboard project.

## [Beta v0.2.1] - 2025-05-11

### Added
- Subtle borders to Analysis section buttons
- Enhanced error handling for DOM operations
- Improved test procedure documentation
- Detailed deployment guide for v0.2.1
- S2 geospatial indexing integration with Zarr store
- Validation reports linking to area reports via S2 cells
- New API endpoints for S2 cell-based report linking
- Click-through functionality for ML overlay to improve marker interaction
- Persistent opacity settings for ML overlay via localStorage
- Header collapse state detection for better UI coordination
- Safer DOM element removal with visibility transitions
- Event-based coordination between UI components

### Changed
- Updated version number in package.json
- Improved ML overlay creation process with gradient background and subtle grid pattern
- Enhanced sidebar management and event handling with custom events
- Enhanced Reports Overview with S2 region indicators
- Object Report view now displays linked validation reports
- Centralized panel management in Sidebar component
- Improved component cleanup processes to prevent memory leaks
- Enhanced map resize handling when toggling panels

### Fixed
- Components/Detection sidebar functionality with proper event handling
- ML overlay opacity control with smoother transitions
- OBJECT DETECTION badge visibility with correct z-index (2000)
- Analytics panel closing when opening Components panel
- Sidebar panel management for better user experience
- Navigation between linked reports
- Header collapse state detection and badge positioning
- Map container resizing during sidebar transitions
- DOM element cleanup to prevent ghost elements
- Performance issues when switching between detection modes

## [Beta v0.2] - 2025-04-26

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
- Updated docs for Beta v0.2

### Fixed
- 3D map state preservation between views
- Map resizing when toggling sidebars
- Header spacing consistency
- Navigation event handling
- Gemini API integration issues
- Removed debug console.log statements

## [Beta v0.1] - 2025-02-15

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