import { cn } from "@/lib/utils";
import React from "react";

interface AuroraBackgroundProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export const AuroraBackground: React.FC<AuroraBackgroundProps> = ({
  className,
  children,
  ...props
}) => {
  return (
    <div
      className={cn(
        "relative flex flex-col min-h-screen items-center justify-center bg-[#0a0a0f] text-white overflow-hidden",
        className
      )}
      {...props}
    >
      {/* Aurora gradient blobs */}
      <div className="absolute inset-0 overflow-hidden">
        <div
          className="absolute -top-[40%] -left-[20%] w-[80%] h-[80%] rounded-full opacity-20 blur-[120px] animate-aurora"
          style={{
            background:
              "radial-gradient(ellipse at center, #76B900 0%, transparent 70%)",
          }}
        />
        <div
          className="absolute -bottom-[30%] -right-[20%] w-[70%] h-[70%] rounded-full opacity-15 blur-[120px] animate-aurora"
          style={{
            background:
              "radial-gradient(ellipse at center, #00b4d8 0%, transparent 70%)",
            animationDelay: "-5s",
          }}
        />
        <div
          className="absolute top-[20%] right-[10%] w-[50%] h-[50%] rounded-full opacity-10 blur-[100px] animate-aurora"
          style={{
            background:
              "radial-gradient(ellipse at center, #7c3aed 0%, transparent 70%)",
            animationDelay: "-10s",
          }}
        />
      </div>

      {/* Noise texture overlay */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />

      {/* Content */}
      <div className="relative z-10 w-full">{children}</div>
    </div>
  );
};

