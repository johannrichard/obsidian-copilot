import React, { useState, useEffect } from "react";
import { useSettingsValue, updateSetting } from "../../model";
import { SettingItem } from "@/components/ui/setting-item";

export const VectorDBSettings: React.FC = () => {
  const settings = useSettingsValue();
  const [showCloudflareSettings, setShowCloudflareSettings] = useState(false);
  const [showQdrantSettings, setShowQdrantSettings] = useState(false);

  const vectorDbOptions = [
    { value: "orama", label: "Orama (Default)" },
    { value: "cloudflare", label: "Cloudflare Vectorize (Cloud-based)" },
    { value: "qdrant", label: "Qdrant (Self-hosted/Cloud)" },
  ];

  const handleVectorDbTypeChange = (value: string) => {
    updateSetting("vectorDbType", value);
  };

  // Update visibility of Cloudflare and Qdrant settings when provider changes
  useEffect(() => {
    setShowCloudflareSettings(settings.vectorDbType === "cloudflare");
    setShowQdrantSettings(settings.vectorDbType === "qdrant");
  }, [settings.vectorDbType]);

  return (
    <div className="space-y-4">
      <section>
        <div className="text-xl font-bold mb-3">Vector Database</div>
        <SettingItem
          type="select"
          title="Vector Database Provider"
          description="Select the vector database provider to use for storing embeddings"
          value={settings.vectorDbType || "orama"}
          onChange={handleVectorDbTypeChange}
          options={vectorDbOptions}
        />

        {showCloudflareSettings && (
          <div className="mt-4 space-y-4 border border-border p-4 rounded-md">
            <div className="text-lg font-medium">Cloudflare Vectorize Settings</div>

            <SettingItem
              type="password"
              title="Cloudflare API Token"
              description="Your Cloudflare API token with Vectorize permissions"
              value={settings.cloudflareApiToken}
              onChange={(value) => updateSetting("cloudflareApiToken", value)}
              placeholder="Enter your Cloudflare API token"
            />

            <SettingItem
              type="text"
              title="Cloudflare Account ID"
              description="Your Cloudflare account ID"
              value={settings.cloudflareAccountId}
              onChange={(value) => updateSetting("cloudflareAccountId", value)}
              placeholder="Enter your Cloudflare account ID"
            />

            <div className="text-sm text-muted mt-2">
              <p>To use Cloudflare Vectorize, you need to:</p>
              <ol className="list-decimal pl-5 mt-2 space-y-1">
                <li>Create a Cloudflare account if you don&apos;t have one</li>
                <li>Enable Vectorize in your Cloudflare account</li>
                <li>Create an API token with Vectorize permissions</li>
                <li>Find your Account ID in the Cloudflare dashboard</li>
              </ol>
              <p className="mt-2">
                <a
                  href="https://developers.cloudflare.com/vectorize/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent hover:underline"
                >
                  Learn more about Cloudflare Vectorize
                </a>
              </p>
            </div>
          </div>
        )}

        {showQdrantSettings && (
          <div className="mt-4 space-y-4 border border-border p-4 rounded-md">
            <div className="text-lg font-medium">Qdrant Settings</div>

            <SettingItem
              type="text"
              title="Qdrant Url"
              description="The protocol, hostname or IP address, and port of your Qdrant server (e.g. http://localhost:6333)"
              value={settings.qdrantUrl}
              onChange={(value) => updateSetting("qdrantUrl", value)}
              placeholder="http://localhost:6333"
            />

            <SettingItem
              type="password"
              title="Qdrant API Key"
              description="The API key for your Qdrant instance (if required)"
              value={settings.qdrantApiKey}
              onChange={(value) => updateSetting("qdrantApiKey", value)}
              placeholder="Enter your Qdrant API key"
            />

            <SettingItem
              type="text"
              title="Qdrant Collection Name"
              description="The name of the Qdrant collection to use (Useful if you want to use finer grained access control. Leave empty to let Copilot manage the creation of the collection). Ensure your API key has write access to this collection."
              value={settings.qdrantCollectionName}
              onChange={(value) => updateSetting("qdrantCollectionName", value)}
              placeholder="copilot-index"
            />

            <div className="text-sm text-muted mt-2">
              <p>To use Qdrant, you need to:</p>
              <ol className="list-decimal pl-5 mt-2 space-y-1">
                <li>Install and run Qdrant locally or in the cloud.</li>
                <li>
                  If using a remote Qdrant instance, ensure it&apos;s accessible from your network.
                </li>
                <li>If your Qdrant instance requires an API key, provide it above.</li>
              </ol>
              <p className="mt-2">
                <a
                  href="https://qdrant.tech/documentation/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent hover:underline"
                >
                  Learn more about Qdrant
                </a>
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
