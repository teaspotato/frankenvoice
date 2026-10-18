import { paramActive, resolveParams } from '../core/node';
import type { Json } from '../core/types';
import { registry } from '../nodes';
import { activeStages, STAGES, stageParams, stageType, type PipelineConfig, type StageId } from '../pipeline';

/** Choose a stage's implementation and edit its params. */
export function StagePanel({
  config,
  stage,
  state,
  disabled,
  onChange,
}: {
  config: PipelineConfig;
  stage: StageId;
  state?: Record<string, Json>;
  disabled: boolean;
  onChange(c: PipelineConfig): void;
}) {
  const def = STAGES.find((s) => s.id === stage)!;
  if (!activeStages(config).some((s) => s.id === stage)) {
    return (
      <>
        <h3>{def.label}</h3>
        <p className="muted">Not used: the TTS stage plays audio itself with the current setting.</p>
      </>
    );
  }
  const node = registry[stageType(config, stage)];
  const values = resolveParams(node, stageParams(config, stage));
  const sc = config[stage];
  const setVariant = (variant: string) => onChange({ ...config, [stage]: { ...sc, variant } });
  const setParam = (key: string, v: Json) =>
    onChange({ ...config, [stage]: { ...sc, params: { ...sc.params, [sc.variant]: { ...stageParams(config, stage), [key]: v } } } });
  const variants = Object.keys(def.variants);
  return (
    <>
      <h3>
        {def.label} <small>{node.type}</small>
      </h3>
      <p className="muted">{node.description}</p>
      {disabled && <p className="locked-note">Locked while running. Stop to change settings.</p>}
      <fieldset className="params" disabled={disabled}>
      {variants.length > 1 && (
        <label className="param">
          <span>Implementation</span>
          <select value={sc.variant} onChange={(e) => setVariant(e.target.value)}>
            {variants.map((v) => (
              <option key={v} value={v}>
                {v} ({registry[def.variants[v]].title})
              </option>
            ))}
          </select>
        </label>
      )}
      {node.params
        .filter((p) => paramActive(p, values) && !def.managed?.includes(p.key))
        .map((p) => {
          const v = values[p.key];
          const set = (x: Json) => setParam(p.key, x);
          return (
            <label key={p.key} className="param">
              <span>{p.label}</span>
              {p.type === 'boolean' ? (
                <input type="checkbox" checked={!!v} onChange={(e) => set(e.target.checked)} />
              ) : p.type === 'select' ? (
                <select value={String(v)} onChange={(e) => set(e.target.value)}>
                  {p.options!.map((o) => (
                    <option key={o}>{o}</option>
                  ))}
                </select>
              ) : p.type === 'text' ? (
                <textarea rows={5} value={String(v)} onChange={(e) => set(e.target.value)} />
              ) : (
                <input
                  type={p.type === 'number' ? 'number' : 'text'}
                  value={String(v)}
                  onChange={(e) => set(p.type === 'number' ? Number(e.target.value) : e.target.value)}
                />
              )}
              {p.description && <small className="muted">{p.description}</small>}
            </label>
          );
        })}
      </fieldset>
      {def.managed && <p className="muted">Browser echo cancellation, noise suppression and gain are set by the Noise / echo stage.</p>}
      {state && (
        <>
          <h4>State</h4>
          <pre className="state">{JSON.stringify(state, null, 2)}</pre>
        </>
      )}
    </>
  );
}
