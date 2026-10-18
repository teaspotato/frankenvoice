import { useEffect, useState } from 'react';

interface Setting {
  name: string;
  group: string;
  label: string;
  secret?: boolean;
  default?: string;
  hint?: string;
  source: 'ui' | 'env' | 'default' | 'unset';
  set: boolean;
  value: string | null;
  preview: string | null;
}

const SOURCE: Record<Setting['source'], string> = { ui: 'set here', env: 'from env', default: 'default', unset: 'not set' };

/**
 * Proxy settings: API keys, base URLs, storage. Values are saved by the proxy
 * (not in the browser); keys are write-only and never sent back to the page.
 */
export function SettingsPanel({ onClose, onSaved }: { onClose(): void; onSaved(): void }) {
  const [settings, setSettings] = useState<Setting[] | null>(null);
  const [file, setFile] = useState('');
  const [edits, setEdits] = useState<Record<string, string | null>>({});
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  const apply = (r: { file: string; settings: Setting[] }) => {
    setFile(r.file);
    setSettings(r.settings);
  };

  useEffect(() => {
    fetch('/api/settings')
      .then(async (r) => (r.ok ? apply(await r.json()) : setError(`proxy answered ${r.status}`)))
      .catch(() => setError('proxy not reachable: run `pnpm dev`'));
  }, []);

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      const r = await fetch('/api/settings', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(edits) });
      const body = await r.json();
      if (!r.ok) throw new Error(body.error ?? `HTTP ${r.status}`);
      apply(body);
      setEdits({});
      onSaved();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setSaving(false);
    }
  };

  const groups = [...new Set((settings ?? []).map((s) => s.group))];
  const dirty = Object.keys(edits).length > 0;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal settings" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Settings</h2>
          <button className="link" onClick={onClose}>
            close
          </button>
        </div>
        <p className="muted">
          Saved by the local proxy{file ? ` in ${file}` : ''}, never in the browser. Values set here override <code>.env</code>; clear one to fall back.
          Keys are write-only: the page only sees whether one is set.
        </p>
        {error && <p className="warn">{error}</p>}
        {groups.map((g) => (
          <fieldset key={g}>
            <legend>{g}</legend>
            {settings!
              .filter((s) => s.group === g)
              .map((s) => {
                const edited = s.name in edits;
                const v = edits[s.name];
                return (
                  <label key={s.name} className="setting" title={s.name}>
                    <span>{s.label}</span>
                    <input
                      type={s.secret ? 'password' : 'text'}
                      autoComplete="off"
                      spellCheck={false}
                      value={edited ? (v ?? '') : s.secret ? '' : s.source === 'ui' || s.source === 'env' ? (s.value ?? '') : ''}
                      placeholder={s.secret ? (s.set ? `${s.preview} (type to replace)` : 'not set') : (s.default ?? s.hint ?? '')}
                      onChange={(e) => setEdits({ ...edits, [s.name]: e.target.value })}
                    />
                    <span className={`source src-${edited ? 'edited' : s.source}`}>{edited ? (v === null ? 'will clear' : 'unsaved') : SOURCE[s.source]}</span>
                    {s.source === 'ui' && !edited ? (
                      <button className="link" onClick={() => setEdits({ ...edits, [s.name]: null })} title="Remove the value set here">
                        clear
                      </button>
                    ) : (
                      <span />
                    )}
                    {s.hint && <small className="muted">{s.hint}</small>}
                  </label>
                );
              })}
          </fieldset>
        ))}
        <div className="modal-foot">
          <button onClick={() => setEdits({})} disabled={!dirty || saving}>
            Discard
          </button>
          <button className="primary" onClick={save} disabled={!dirty || saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
