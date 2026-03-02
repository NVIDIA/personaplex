import { FC, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
};

export const TextDisplay: FC<TextDisplayProps> = ({
  containerRef,
}) => {
  const { text } = useServerText();
  const currentIndex = text.length - 1;
  const prevScrollTop = useRef(0);

  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [text]);

  return (
    <div className="h-full w-full max-w-full max-h-full p-2 space-y-1">
      {text.map((t, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
          className={`inline ${i === currentIndex
              ? "text-white font-medium"
              : "text-white/60 font-normal"
            }`}
        >
          {t}
        </motion.span>
      ))}
      {text.length > 0 && (
        <motion.span
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
          className="inline-block w-2 h-4 bg-[#76B900] rounded-sm ml-1 align-middle"
        />
      )}
    </div>
  );
};
