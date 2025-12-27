#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataScraper.py — FRAXplus collector (US 4 groups) +  ML (not run unless --train 1).

Rules kept:
- NO in-memory rows list
- If results are not detected (timeout/failure), write ALL THREE outputs as null/NaN:
  t_score, mof_risk, hip_risk -> NaN
- Open site once, set fixed dropdowns once, avoid Clear.
- Write CSV row-by-row as we go (append mode)
- also write XLSX at the end
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    ElementNotInteractableException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
)

from webdriver_manager.chrome import ChromeDriverManager

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


FRAXPLUS_URL = "https://www.fraxplus.org/calculation-tool"
CONTINENT = "North America"
US_GROUPS = ["US (Caucasian)", "US (Black)", "US (Asian)", "US (Hispanic)"]
BMI_UNITS = "kg / cm"
DEFAULT_SCANNER = "Hologic"

SELECTORS = {
    # Numeric inputs
    "age":    (By.NAME, "age"),
    "weight": (By.NAME, "bmi.weight"),
    "height": (By.NAME, "bmi.height"),
    "bmd":    (By.NAME, "femoralNeckBMD.value"),

    # Sex radio
    "sex_f": (By.ID, "sri-F"),
    "sex_m": (By.ID, "sri-M"),

    # Checkboxes
    "previousFracture":      (By.ID, "chx-previousFracture"),
    "parentFracturedHip":    (By.ID, "chx-parentFracturedHip"),
    "smoking":               (By.ID, "chx-smoking"),
    "glucocorticoids":       (By.ID, "chx-glucocorticoids"),
    "rheumatoidArthritis":   (By.ID, "chx-rheumatoidArthritis"),
    "secondaryOsteoporosis": (By.ID, "chx-secondaryOsteoporosis"),
    "alcohol":               (By.ID, "chx-alcohol"),

    # Buttons
    "calculate_btn": (By.XPATH, "//button[normalize-space()='Calculate']"),
}

CSV_COLUMNS = [
    "continent", "bmi_units", "scanner",
    "us_group", "age", "sex_female", "weight_kg", "height_cm", "bmi", "bmd",
    "previousFracture", "parentFracturedHip", "smoking", "glucocorticoids",
    "rheumatoidArthritis", "secondaryOsteoporosis", "alcohol",
    "t_score", "mof_risk", "hip_risk",
]


# ------------------- Browser helpers -------------------

def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(90)
    return driver


def safe_scroll_into_view(driver, element):
    driver.execute_script(
        "arguments[0].scrollIntoView({block:'center'});", element)


def safe_click(driver, locator, wait_s: float):
    el = WebDriverWait(driver, wait_s).until(
        EC.element_to_be_clickable(locator))
    safe_scroll_into_view(driver, el)
    try:
        el.click()
    except (ElementClickInterceptedException, ElementNotInteractableException):
        driver.execute_script("arguments[0].click();", el)
    return el


def clear_and_type(driver, locator, value, wait_s: float):
    el = WebDriverWait(driver, wait_s).until(
        EC.presence_of_element_located(locator))
    safe_scroll_into_view(driver, el)
    el.click()
    el.send_keys(Keys.COMMAND + "a")
    el.send_keys(Keys.BACKSPACE)
    el.send_keys(str(value))
    return el


def save_debug_artifacts(driver, prefix="fraxplus_debug"):
    png = f"{prefix}.png"
    html = f"{prefix}.html"
    driver.save_screenshot(png)
    with open(html, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print(f"Saved screenshot: {png}")
    print(f"Saved HTML:       {html}")


def try_accept_cookies(driver):
    for txt in ["Accept", "Accept all", "I agree", "OK", "Agree"]:
        try:
            btn = driver.find_element(
                By.XPATH, f"//button[contains(., '{txt}')]")
            btn.click()
            return
        except Exception:
            pass


# ------------------- React-select (ID-independent) -------------------

def get_react_combobox_inputs(driver) -> List:
    return driver.find_elements(
        By.CSS_SELECTOR,
        "input[role='combobox'][id^='react-select-'][id$='-input']"
    )


def react_select_pick_by_index(driver, index: int, option_text: str, wait_s: float):
    """
    Index mapping:
      0: Continent
      1: Country/Model
      2: BMI Units
      3: Scanner
    """
    def _enough(d):
        elems = get_react_combobox_inputs(d)
        return elems if len(elems) >= (index + 1) else False

    combos = WebDriverWait(driver, wait_s).until(_enough)
    inp = combos[index]

    safe_scroll_into_view(driver, inp)
    try:
        inp.click()
    except Exception:
        driver.execute_script("arguments[0].click();", inp)

    inp.send_keys(Keys.COMMAND + "a")
    inp.send_keys(Keys.BACKSPACE)
    inp.send_keys(option_text)

    opt_xpath = f"//div[@role='option' and normalize-space()='{option_text}']"
    opt = WebDriverWait(driver, wait_s).until(
        EC.element_to_be_clickable((By.XPATH, opt_xpath)))
    safe_scroll_into_view(driver, opt)
    try:
        opt.click()
    except Exception:
        driver.execute_script("arguments[0].click();", opt)

    time.sleep(0.04)


# ------------------- Checkbox robust click -------------------

def set_checkbox(driver, locator, checked: bool, wait_s: float):
    el = WebDriverWait(driver, wait_s).until(
        EC.presence_of_element_located(locator))
    safe_scroll_into_view(driver, el)

    for _ in range(3):
        try:
            current = el.is_selected()
            break
        except StaleElementReferenceException:
            el = WebDriverWait(driver, wait_s).until(
                EC.presence_of_element_located(locator))
            safe_scroll_into_view(driver, el)
    else:
        current = el.is_selected()

    if current == checked:
        return

    el_id = el.get_attribute("id")

    try:
        WebDriverWait(driver, wait_s).until(
            EC.element_to_be_clickable(locator))
        try:
            el.click()
            return
        except (ElementNotInteractableException, ElementClickInterceptedException):
            pass
    except TimeoutException:
        pass

    if el_id:
        try:
            lab = driver.find_element(By.CSS_SELECTOR, f"label[for='{el_id}']")
            safe_scroll_into_view(driver, lab)
            try:
                lab.click()
            except Exception:
                driver.execute_script("arguments[0].click();", lab)
            return
        except Exception:
            pass

    driver.execute_script("arguments[0].click();", el)


# ------------------- Results parsing -------------------

def _normalize_minuses(txt: str) -> str:
    return (txt
            .replace("\u2212", "-")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-"))


def extract_results_strict(driver) -> Optional[Dict[str, float]]:
    try:
        body_text = _normalize_minuses(
            driver.find_element(By.TAG_NAME, "body").text)
    except Exception:
        return None

    mof_m = re.search(
        r"Major\s+osteoporotic(?:\s+fracture)?\s*([\d.]+)\s*%",
        body_text, flags=re.IGNORECASE
    )
    hip_m = re.search(
        r"Hip\s+Fracture\s*([\d.]+)\s*%",
        body_text, flags=re.IGNORECASE
    )
    if not (mof_m and hip_m):
        return None

    t_m = re.search(
        r"T-?score\s*:\s*(-?\d+(?:\.\d+)?)",
        body_text, flags=re.IGNORECASE
    )

    return {
        "t_score": float(t_m.group(1)) if t_m else np.nan,
        "mof_risk": float(mof_m.group(1)),
        "hip_risk": float(hip_m.group(1)),
    }


def result_signature(res: Optional[Dict[str, float]]) -> Optional[Tuple[float, float, float]]:
    if not res:
        return None
    return (res.get("t_score", np.nan), res.get("mof_risk", np.nan), res.get("hip_risk", np.nan))


def wait_for_new_results(driver, prev_sig: Optional[Tuple[float, float, float]], timeout_s: float) -> Optional[Dict[str, float]]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        res = extract_results_strict(driver)
        if res:
            sig = result_signature(res)
            if prev_sig is None or sig != prev_sig:
                return res
        time.sleep(0.10)
    return None


# ------------------- Patient sampling -------------------

@dataclass
class Patient:
    age: int
    sex: str
    weight_kg: float
    height_cm: float
    bmd: float
    previousFracture: int
    parentFracturedHip: int
    smoking: int
    glucocorticoids: int
    rheumatoidArthritis: int
    secondaryOsteoporosis: int
    alcohol: int


def sample_patient(rng: np.random.Generator) -> Patient:
    age = int(rng.integers(40, 91))
    sex = "F" if rng.random() < 0.55 else "M"

    height_cm = float(
        np.clip(rng.normal(163 if sex == "F" else 176, 7.5), 145, 200))
    bmi = float(np.clip(rng.normal(26, 4.5), 18.5, 40.0))
    weight_kg = float(np.clip(bmi * (height_cm / 100.0) ** 2, 40, 160))

    bmd = float(np.clip(rng.normal(0.80, 0.14) -
                0.003 * max(age - 55, 0), 0.40, 1.20))

    return Patient(
        age=age,
        sex=sex,
        weight_kg=round(weight_kg, 1),
        height_cm=round(height_cm, 1),
        bmd=round(bmd, 3),
        previousFracture=int(rng.random() < 0.18),
        parentFracturedHip=int(rng.random() < 0.12),
        smoking=int(rng.random() < 0.22),
        glucocorticoids=int(rng.random() < 0.07),
        rheumatoidArthritis=int(rng.random() < 0.05),
        secondaryOsteoporosis=int(rng.random() < 0.08),
        alcohol=int(rng.random() < 0.15),
    )


# ------------------- One-time setup + per-row calc -------------------

def open_and_setup(driver, scanner: str, wait_s: float):
    driver.get(FRAXPLUS_URL)
    try_accept_cookies(driver)

    # fixed dropdowns once
    react_select_pick_by_index(driver, 0, CONTINENT, wait_s=wait_s)
    react_select_pick_by_index(driver, 2, BMI_UNITS, wait_s=wait_s)
    react_select_pick_by_index(driver, 3, scanner, wait_s=wait_s)


def fraxplus_fill_and_calc(driver, us_group: str, patient: Patient,
                           wait_s: float, result_timeout_s: float,
                           debug: bool = False) -> Dict[str, float]:
    """
    If results not detected: return all NaNs for t_score/mof/hip.
    """
    # Set country/model each row
    react_select_pick_by_index(driver, 1, us_group, wait_s=wait_s)

    # Sex
    safe_click(driver, SELECTORS["sex_f"] if patient.sex ==
               "F" else SELECTORS["sex_m"], wait_s=wait_s)

    # Numeric inputs
    clear_and_type(driver, SELECTORS["age"], patient.age, wait_s=wait_s)
    clear_and_type(driver, SELECTORS["weight"],
                   patient.weight_kg, wait_s=wait_s)
    clear_and_type(driver, SELECTORS["height"],
                   patient.height_cm, wait_s=wait_s)
    clear_and_type(driver, SELECTORS["bmd"], patient.bmd, wait_s=wait_s)

    # Checkboxes
    set_checkbox(driver, SELECTORS["previousFracture"], bool(
        patient.previousFracture), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["parentFracturedHip"], bool(
        patient.parentFracturedHip), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["smoking"],
                 bool(patient.smoking), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["glucocorticoids"], bool(
        patient.glucocorticoids), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["rheumatoidArthritis"], bool(
        patient.rheumatoidArthritis), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["secondaryOsteoporosis"], bool(
        patient.secondaryOsteoporosis), wait_s=wait_s)
    set_checkbox(driver, SELECTORS["alcohol"],
                 bool(patient.alcohol), wait_s=wait_s)

    prev_res = extract_results_strict(driver)
    prev_sig = result_signature(prev_res)

    # Calculate
    safe_click(driver, SELECTORS["calculate_btn"], wait_s=wait_s)

    res = wait_for_new_results(
        driver, prev_sig=prev_sig, timeout_s=result_timeout_s)

    # REQUIRED: if res is None -> all 3 null
    if res is None:
        if debug:
            save_debug_artifacts(driver, prefix="fraxplus_debug_no_results")
        return {"t_score": np.nan, "mof_risk": np.nan, "hip_risk": np.nan}

    return res


def fraxplus_calc_with_retry(driver, us_group: str, patient: Patient,
                             wait_s: float, result_timeout_s: float,
                             debug: bool, retries: int = 1) -> Dict[str, float]:
    last_err = None
    for k in range(retries + 1):
        try:
            return fraxplus_fill_and_calc(
                driver, us_group, patient,
                wait_s=wait_s,
                result_timeout_s=result_timeout_s,
                debug=debug
            )
        except (ElementNotInteractableException, ElementClickInterceptedException,
                StaleElementReferenceException, TimeoutException, RuntimeError) as e:
            last_err = e
            if debug:
                print(f"[retry {k+1}/{retries}] {type(e).__name__}: {e}")
                save_debug_artifacts(
                    driver, prefix=f"fraxplus_debug_retry{k+1}")
            time.sleep(0.30)

    # If retries exhausted, still write NaNs so run continues
    if debug and last_err:
        print(
            f"[final] returning NaNs after error: {type(last_err).__name__}: {last_err}")
    return {"t_score": np.nan, "mof_risk": np.nan, "hip_risk": np.nan}


# ------------------- Row-by-row CSV writer -------------------

def init_csv(out_csv: str):
    # overwrite file and write header
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(out_csv, index=False)


def append_row_csv(out_csv: str, row: Dict):
    # ensure consistent column order
    df1 = pd.DataFrame([row], columns=CSV_COLUMNS)
    df1.to_csv(out_csv, mode="a", header=False, index=False)


# ------------------- Dataset collection (streaming to CSV) -------------------

def collect_dataset_to_csv(n_per_group: int, seed: int, out_csv: str,
                           headless: bool, debug: bool,
                           scanner: str, wait_s: float, result_timeout_s: float,
                           also_write_xlsx_end: bool = False):
    rng = np.random.default_rng(seed)
    driver = make_driver(headless=headless)

    init_csv(out_csv)

    try:
        open_and_setup(driver, scanner=scanner, wait_s=wait_s)

        for group in US_GROUPS:
            for _ in tqdm(range(n_per_group), desc=f"Collecting {group}"):
                patient = sample_patient(rng)

                y = fraxplus_calc_with_retry(
                    driver, us_group=group, patient=patient,
                    wait_s=wait_s, result_timeout_s=result_timeout_s,
                    debug=debug, retries=1
                )

                bmi_val = float(patient.weight_kg) / \
                    ((float(patient.height_cm) / 100.0) ** 2)

                row = {
                    "continent": CONTINENT,
                    "bmi_units": BMI_UNITS,
                    "scanner": scanner,

                    "us_group": group,
                    "age": patient.age,
                    "sex_female": 1 if patient.sex == "F" else 0,
                    "weight_kg": patient.weight_kg,
                    "height_cm": patient.height_cm,
                    "bmi": round(bmi_val, 2),
                    "bmd": patient.bmd,
                    "previousFracture": patient.previousFracture,
                    "parentFracturedHip": patient.parentFracturedHip,
                    "smoking": patient.smoking,
                    "glucocorticoids": patient.glucocorticoids,
                    "rheumatoidArthritis": patient.rheumatoidArthritis,
                    "secondaryOsteoporosis": patient.secondaryOsteoporosis,
                    "alcohol": patient.alcohol,

                    "t_score": y.get("t_score", np.nan),
                    "mof_risk": y.get("mof_risk", np.nan),
                    "hip_risk": y.get("hip_risk", np.nan),
                }

                append_row_csv(out_csv, row)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(f"\nSaved CSV incrementally: {out_csv}")

    if also_write_xlsx_end:
        df = pd.read_csv(out_csv)
        xlsx_path = out_csv.replace(".csv", ".xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Saved Excel (end): {xlsx_path}")


# -------------------  ML (kept, not run unless --train 1) -------------------

def train_models(df: pd.DataFrame, out_prefix: str):
    X = df.drop(columns=["mof_risk", "hip_risk"])
    X = pd.get_dummies(X, columns=["us_group"], drop_first=False)

    y_mof = df["mof_risk"].astype(float).values
    y_hip = df["hip_risk"].astype(float).values

    X_train, X_test, y_mof_train, y_mof_test, y_hip_train, y_hip_test = train_test_split(
        X, y_mof, y_hip, test_size=0.2, random_state=42
    )

    mof_model = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=800,
        min_samples_leaf=40, l2_regularization=0.2,
        random_state=42, early_stopping=True
    )
    hip_model = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=800,
        min_samples_leaf=40, l2_regularization=0.2,
        random_state=42, early_stopping=True
    )

    print("\nTraining MOF model...")
    mof_model.fit(X_train, y_mof_train)
    print("Training Hip model...")
    hip_model.fit(X_train, y_hip_train)

    mof_pred = mof_model.predict(X_test)
    hip_pred = hip_model.predict(X_test)

    print("\n=== PERFORMANCE (holdout) ===")
    print(
        f"MOF  MAE: {mean_absolute_error(y_mof_test, mof_pred):.3f} | R2: {r2_score(y_mof_test, mof_pred):.4f}")
    print(
        f"Hip  MAE: {mean_absolute_error(y_hip_test, hip_pred):.3f} | R2: {r2_score(y_hip_test, hip_pred):.4f}")

    joblib.dump(mof_model, f"{out_prefix}_mof.joblib")
    joblib.dump(hip_model, f"{out_prefix}_hip.joblib")
    with open(f"{out_prefix}_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    print(
        f"\nSaved:\n - {out_prefix}_mof.joblib\n - {out_prefix}_hip.joblib\n - {out_prefix}_columns.json")


# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_group", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="fraxplus_us2_dataset.csv")
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--scanner", type=str, default=DEFAULT_SCANNER)

    ap.add_argument("--wait_s", type=float, default=10.0)
    ap.add_argument("--result_timeout_s", type=float, default=30.0)

    # if you want an xlsx at the end (optional)
    ap.add_argument("--out_xlsx_end", type=int, default=0)

    # ML kept but NOT run unless --train 1
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--out_prefix", type=str, default="fraxplus_us4_surrogate")

    args = ap.parse_args()

    collect_dataset_to_csv(
        n_per_group=args.n_per_group,
        seed=args.seed,
        out_csv=args.out_csv,
        headless=(args.headless == 1),
        debug=(args.debug == 1),
        scanner=args.scanner,
        wait_s=args.wait_s,
        result_timeout_s=args.result_timeout_s,
        also_write_xlsx_end=(args.out_xlsx_end == 1),
    )

    # ML is not run by default; if you do run it, it reads the CSV you wrote.
    if args.train == 1:
        df = pd.read_csv(args.out_csv)
        if len(df) < 40:
            print("\nNot training: dataset too small. Increase --n_per_group.")
        else:
            train_models(df, args.out_prefix)


if __name__ == "__main__":
    main()

# python3 Desktop/frax1.py --n_per_group 50 --out_xlsx_end 1
