import { createContext, useContext } from "react";
import en from "./en.json";
import ko from "./ko.json";

export type Language = "en" | "ko";

const translations: Record<Language, typeof en> = { en, ko };

export type I18nContextType = {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
};

/**
 * Get a nested value from an object using a dot-separated key path.
 */
function getNestedValue(obj: Record<string, unknown>, keyPath: string): string {
  const keys = keyPath.split(".");
  let current: unknown = obj;
  for (const k of keys) {
    if (current === null || current === undefined || typeof current !== "object") {
      return keyPath;
    }
    current = (current as Record<string, unknown>)[k];
  }
  return typeof current === "string" ? current : keyPath;
}

export function translate(language: Language, key: string): string {
  return getNestedValue(
    translations[language] as unknown as Record<string, unknown>,
    key,
  );
}

export const I18nContext = createContext<I18nContextType>({
  language: "en",
  setLanguage: () => {},
  t: (key: string) => translate("en", key),
});

export function useI18n(): I18nContextType {
  return useContext(I18nContext);
}
