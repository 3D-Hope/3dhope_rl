# Migration to Vite + TypeScript + React Router

## What Changed

The frontend has been migrated from Create React App to Vite, with TypeScript and React Router.

## New Structure

### Components (TypeScript)
- `src/components/LandingPage.tsx` - Landing page with gallery
- `src/components/Generator.tsx` - Main scene generation interface
- `src/components/PolygonCanvas.tsx` - Polygon drawing canvas
- `src/components/ThreeJSGLBViewer.tsx` - 3D model viewer

### Utilities
- `src/utils/api.ts` - API client functions
- `src/types/index.ts` - TypeScript type definitions

### Routing
- `/` - Landing page
- `/generate` - Scene generator

## Setup Instructions

1. **Install dependencies:**
   ```bash
   cd app/frontend
   npm install
   ```

2. **Create `.env` file (optional):**
   ```bash
   echo "VITE_API_URL=http://localhost:8000" > .env
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Build for production:**
   ```bash
   npm run build
   ```

## Old Files to Remove

After confirming everything works, you can remove:
- `src/App.js` (replaced by `src/App.tsx`)
- `src/index.js` (replaced by `src/main.tsx`)
- `public/index.html` (moved to root `index.html`)

## Key Differences

1. **Environment Variables**: Use `import.meta.env.VITE_API_URL` instead of `process.env.REACT_APP_API_URL`
2. **Imports**: All components are now TypeScript (.tsx)
3. **Routing**: Uses React Router instead of conditional rendering
4. **Build**: Uses Vite instead of react-scripts
