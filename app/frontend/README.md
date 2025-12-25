# Auto-Decor Frontend

A modern React + TypeScript + Vite application for AI-powered 3D scene generation.

## Features

- 🎨 Custom polygon floor plan drawing
- 🤖 AI-powered scene generation
- 📐 Interactive 3D visualization
- 🚀 Fast development with Vite
- 💪 TypeScript for type safety
- 🧩 Component-based architecture

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── LandingPage.tsx
│   ├── Generator.tsx
│   ├── PolygonCanvas.tsx
│   └── ThreeJSGLBViewer.tsx
├── types/              # TypeScript type definitions
│   └── index.ts
├── utils/              # Utility functions
│   └── api.ts
├── assets/             # Static assets (images)
├── App.tsx             # Main app component with routing
├── App.css             # Styles
├── main.tsx            # Entry point
└── index.css           # Global styles
```

## Environment Variables

Create a `.env` file in the root directory:

```
VITE_API_URL=http://localhost:8000
```

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Three.js** - 3D visualization
- **Axios** - HTTP client
