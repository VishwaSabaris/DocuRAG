import {
  Moon,
  Save,
  Sun,
  X,
} from "lucide-react";

import type {
  AppSettings,
} from "../types/chat";

interface SettingsModalProps {
  open: boolean;
  settings: AppSettings;
  onChange: (
    settings: AppSettings,
  ) => void;
  onClose: () => void;
}

export function SettingsModal({
  open,
  settings,
  onChange,
  onClose,
}: SettingsModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="modal-backdrop">
      <div className="settings-modal">
        <div className="settings-header">
          <div>
            <h2>Settings</h2>
            <p>
              Configure your DocuRAG
              workspace.
            </p>
          </div>

          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close settings"
          >
            <X size={18} />
          </button>
        </div>

        <div className="settings-content">
          <label className="settings-field">
            <span>
              Backend API URL
            </span>

            <input
              value={
                settings.apiBaseUrl
              }
              onChange={(event) =>
                onChange({
                  ...settings,
                  apiBaseUrl:
                    event.target.value,
                })
              }
              placeholder="http://localhost:8000"
            />
          </label>

          <div className="settings-field">
            <span>Appearance</span>

            <div className="theme-options">
              <button
                className={
                  settings.theme ===
                  "dark"
                    ? "theme-option active"
                    : "theme-option"
                }
                onClick={() =>
                  onChange({
                    ...settings,
                    theme: "dark",
                  })
                }
              >
                <Moon size={17} />
                Dark
              </button>

              <button
                className={
                  settings.theme ===
                  "light"
                    ? "theme-option active"
                    : "theme-option"
                }
                onClick={() =>
                  onChange({
                    ...settings,
                    theme: "light",
                  })
                }
              >
                <Sun size={17} />
                Light
              </button>
            </div>
          </div>
        </div>

        <div className="settings-footer">
          <button
            className="primary-button"
            onClick={onClose}
          >
            <Save size={16} />
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
