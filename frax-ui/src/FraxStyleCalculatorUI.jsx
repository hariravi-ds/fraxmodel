import React, { useMemo, useState } from "react";

function Switch({ checked, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={[
        "relative inline-flex h-7 w-14 items-center rounded-full transition-colors",
        checked ? "bg-emerald-500" : "bg-slate-300",
      ].join(" ")}
    >
      <span
        className={[
          "inline-block h-6 w-6 transform rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-7" : "translate-x-1",
        ].join(" ")}
      />
      <span
        className={[
          "absolute right-3 text-xs font-semibold",
          checked ? "text-white" : "text-slate-600",
        ].join(" ")}
      >
        {checked ? "✓" : "×"}
      </span>
    </button>
  );
}

function FieldLabel({ children }) {
  return <div className="text-sm font-semibold text-slate-800">{children}</div>;
}

function Input({ className = "", ...props }) {
  return (
    <input
      {...props}
      className={[
        "h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none",
        "focus:border-red-500 focus:ring-2 focus:ring-red-100",
        className,
      ].join(" ")}
    />
  );
}

function Select({ className = "", children, ...props }) {
  return (
    <select
      {...props}
      className={[
        "h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none",
        "focus:border-red-500 focus:ring-2 focus:ring-red-100",
        className,
      ].join(" ")}
    >
      {children}
    </select>
  );
}

export default function FraxStyleCalculatorUI() {
  const [continent, setContinent] = useState("North America");
  const [country, setCountry] = useState("US (Asian)");

  const [age, setAge] = useState(65);
  const [sex, setSex] = useState("female");
  const [weightKg, setWeightKg] = useState(62);
  const [heightCm, setHeightCm] = useState(160);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [risk, setRisk] = useState({
    previousFracture: false,
    parentHipFracture: false,
    currentSmoking: false,
    glucocorticoids: false,
    rheumatoidArthritis: false,
    secondaryOsteoporosis: false,
    alcohol3Plus: false,
  });

  const [bmdType, setBmdType] = useState("");
  const [bmdValue, setBmdValue] = useState(0.72);

  const bmi = useMemo(() => {
    const hM = Number(heightCm) / 100;
    const w = Number(weightKg);
    if (!hM || !w) return null;
    const v = w / (hM * hM);
    return Number.isFinite(v) ? v : null;
  }, [heightCm, weightKg]);

  const [calculated, setCalculated] = useState(false);

  const onClear = () => {
    setContinent("North America");
    setCountry("US (Asian)");
    setLocalRef("");

    setAge(65);
    setSex("female");
    setWeightKg(62);
    setHeightCm(160);

    setRisk({
      previousFracture: false,
      parentHipFracture: false,
      currentSmoking: false,
      glucocorticoids: false,
      rheumatoidArthritis: false,
      secondaryOsteoporosis: false,
      alcohol3Plus: false,
    });

    setBmdType("");
    setBmdValue(0.72);
    setCalculated(false);
  };

  const onCalculate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = {
        continent,
        country,
        age,
        sex,
        weight_kg: weightKg,
        height_cm: heightCm,
        risk_factors: {
          previousFracture: risk.previousFracture,
          parentFracturedHip: risk.parentHipFracture,
          smoking: risk.currentSmoking,
          glucocorticoids: risk.glucocorticoids,
          rheumatoidArthritis: risk.rheumatoidArthritis,
          secondaryOsteoporosis: risk.secondaryOsteoporosis,
          alcohol: risk.alcohol3Plus,
        },
        femoral_neck: bmdType
          ? { type: bmdType, value: Number(bmdValue) }
          : null,
      };

      const resp = await fetch("/api/frax/calculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json(); // expect { mof: ..., hip: ... }
      setResult(data);
    } catch (err) {
      setError(err?.message || "API failed");
    } finally {
      setLoading(false);
    }
  };

  const ToggleRow = ({ n, title, keyName }) => (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="text-sm text-slate-800">
        <span className="mr-2 font-semibold text-slate-600">{n}.</span>
        {title}
      </div>
      <Switch
        checked={risk[keyName]}
        onChange={(v) => setRisk((r) => ({ ...r, [keyName]: v }))}
        label={title}
      />
    </div>
  );

  return (
    <div className="min-h-screen bg-white">
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="text-sm text-slate-600">
            Please answer the questions below to calculate the ten-year
            probability of fracture with or without BMD.
          </div>

          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-12">
            <div className="md:col-span-4">
              <FieldLabel>Continent</FieldLabel>
              <Select value={continent} disabled>
                <option>North America</option>
                <option>South America</option>
                <option>Europe</option>
                <option>Asia</option>
                <option>Africa</option>
                <option>Oceania</option>
              </Select>
            </div>

            <div className="md:col-span-4">
              <FieldLabel>Country</FieldLabel>
              <Select
                value={country}
                onChange={(e) => setCountry(e.target.value)}
              >
                <option>US (Asian)</option>
                <option>US (Caucasian)</option>
                <option>US (Black)</option>
                <option>US (Hispanic)</option>
              </Select>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2 text-sm text-red-600">
            <span className="font-semibold">About the risk factors</span>
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-red-200 bg-red-50 text-xs font-bold">
              ?
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <form
            onSubmit={onCalculate}
            className="md:col-span-7 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div className="mb-5 text-4xl font-extrabold text-red-600">
              Questionnaire
            </div>

            {/* 1 Age */}
            <div className="mb-4 grid grid-cols-1 items-center gap-3 sm:grid-cols-12">
              <div className="sm:col-span-7 text-sm text-slate-800">
                <span className="mr-2 font-semibold text-slate-600">1.</span>
                Age (between 40 and 90 years)
              </div>
              <div className="sm:col-span-5">
                <Input
                  type="number"
                  min={40}
                  max={90}
                  value={age}
                  onChange={(e) =>
                    setAge(e.target.value === "" ? "" : Number(e.target.value))
                  }
                />
              </div>
            </div>

            {/* 2 Sex */}
            <div className="mb-4 grid grid-cols-1 items-center gap-3 sm:grid-cols-12">
              <div className="sm:col-span-7 text-sm text-slate-800">
                <span className="mr-2 font-semibold text-slate-600">2.</span>
                Sex
              </div>
              <div className="sm:col-span-5 flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="sex"
                    checked={sex === "female"}
                    onChange={() => setSex("female")}
                  />
                  Female
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="sex"
                    checked={sex === "male"}
                    onChange={() => setSex("male")}
                  />
                  Male
                </label>
              </div>
            </div>

            {/* 3 Weight */}
            <div className="mb-4 grid grid-cols-1 items-center gap-3 sm:grid-cols-12">
              <div className="sm:col-span-7 text-sm text-slate-800">
                <span className="mr-2 font-semibold text-slate-600">3.</span>
                Weight
              </div>
              <div className="sm:col-span-5 flex items-center gap-2">
                <span className="text-sm text-slate-500">kg</span>
                <Input
                  type="number"
                  min={20}
                  max={200}
                  value={weightKg}
                  onChange={(e) =>
                    setWeightKg(
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
                  }
                />
              </div>
            </div>

            {/* 4 Height */}
            <div className="mb-4 grid grid-cols-1 items-center gap-3 sm:grid-cols-12">
              <div className="sm:col-span-7 text-sm text-slate-800">
                <span className="mr-2 font-semibold text-slate-600">4.</span>
                Height
              </div>
              <div className="sm:col-span-5 flex items-center gap-2">
                <span className="text-sm text-slate-500">cm</span>
                <Input
                  type="number"
                  min={100}
                  max={220}
                  value={heightCm}
                  onChange={(e) =>
                    setHeightCm(
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
                  }
                />
              </div>
            </div>

            <div className="mb-4 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
              <span className="font-semibold">BMI:</span>{" "}
              {bmi == null ? "—" : bmi.toFixed(1)}
            </div>

            {/* 5-11 Toggles */}
            <div className="mt-2 divide-y divide-slate-200">
              <ToggleRow
                n={5}
                title="Previous Fracture"
                keyName="previousFracture"
              />
              <ToggleRow
                n={6}
                title="Parent Fractured Hip"
                keyName="parentHipFracture"
              />
              <ToggleRow
                n={7}
                title="Current smoking"
                keyName="currentSmoking"
              />
              <ToggleRow
                n={8}
                title="Glucocorticoids"
                keyName="glucocorticoids"
              />
              <ToggleRow
                n={9}
                title="Rheumatoid arthritis"
                keyName="rheumatoidArthritis"
              />
              <ToggleRow
                n={10}
                title="Secondary osteoporosis"
                keyName="secondaryOsteoporosis"
              />
              <ToggleRow
                n={11}
                title="Alcohol 3 or more units/day"
                keyName="alcohol3Plus"
              />
            </div>

            {/* Actions */}
            <div className="mt-6 flex flex-col gap-3 md:hidden">
              <button
                type="submit"
                className="h-12 rounded-xl bg-red-600 text-white shadow-sm transition hover:bg-red-700"
                onClick={onCalculate}
              >
                Calculate
              </button>
              <button
                type="button"
                onClick={onClear}
                className="h-12 rounded-xl border border-red-300 bg-white text-red-600 shadow-sm transition hover:bg-red-50"
              >
                Clear
              </button>
            </div>
          </form>

          {/* Right: BMD + actions */}
          <div className="md:col-span-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-800">
                <span className="mr-2 font-semibold text-slate-600">12.</span>
                Femoral neck BMD
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-12">
              <div className="sm:col-span-7">
                <Select
                  value={bmdType}
                  onChange={(e) => setBmdType(e.target.value)}
                >
                  <option value="">Select BMD</option>
                  <option value="bmd_g_cm2">BMD (g/cm²)</option>
                  <option value="t_score">T-score</option>
                </Select>
              </div>
              <div className="sm:col-span-5">
                <Input
                  type="number"
                  step="0.01"
                  value={bmdValue}
                  onChange={(e) =>
                    setBmdValue(
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
                  }
                />
              </div>
            </div>

            <div className="mt-3 text-xs text-slate-500">
              Tip: pick whether the value is{" "}
              <span className="font-semibold">BMD</span> or{" "}
              <span className="font-semibold">T-score</span>.
            </div>

            <div className="mt-6 hidden gap-3 md:flex">
              <button
                onClick={onCalculate}
                className="h-12 flex-1 rounded-xl bg-red-600 text-white shadow-sm transition hover:bg-red-700"
              >
                Calculate
              </button>
              <button
                type="button"
                onClick={onClear}
                className="h-12 flex-1 rounded-xl border border-red-300 bg-white text-red-600 shadow-sm transition hover:bg-red-50"
              >
                Clear
              </button>
            </div>

            {/* Result panel */}
            <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
              {!calculated ? (
                <div className="text-sm text-slate-600"></div>
              ) : (
                <div className="space-y-2 text-sm text-slate-700">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Age / Sex</span>
                    <span className="font-semibold">
                      {age} / {sex === "female" ? "Female" : "Male"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Weight / Height</span>
                    <span className="font-semibold">
                      {weightKg} kg / {heightCm} cm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">BMI</span>
                    <span className="font-semibold">
                      {bmi == null ? "—" : bmi.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">BMD input</span>
                    <span className="font-semibold">
                      {bmdType
                        ? bmdType === "t_score"
                          ? "T-score"
                          : "BMD"
                        : "Not selected"}{" "}
                      {bmdValue !== "" ? `(${bmdValue})` : ""}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
