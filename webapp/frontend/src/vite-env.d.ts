/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin for hosted builds; empty in dev so the Vite proxy handles /api. */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
