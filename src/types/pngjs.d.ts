declare module "pngjs" {
  // pngjs ships without types; sync API only used in tests + scripts.
  export const PNG: {
    new (opts: { width: number; height: number; colorType?: number }): { data: Buffer };
    sync: {
      write(
        png: { width: number; height: number; data: Buffer },
        opts?: Record<string, unknown>,
      ): Buffer;
    };
  };
}
