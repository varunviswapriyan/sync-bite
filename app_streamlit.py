import os, math, re, json
from datetime import time, datetime
import pandas as pd, requests, streamlit as st
import numpy as np

import json, re, os
from openai import OpenAI

def fetch_menu(name: str, meal: str, date):
    """
    Robustly fetch and normalize a structured menu via OpenAI.
    Guarantees the returned dict has keys: provider, meal, items (list).
    """
    # Fallback skeleton we will *always* return at minimum
    def skeleton(items=None, err=None):
        return {
            "provider": name,
            "meal": meal,
            "items": items if isinstance(items, list) else ([] if err else [{"category": "Menu", "options": []}]),
            **({"error": str(err)} if err else {})
        }

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return skeleton(err="Missing OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)

        prompt = f"""
Return ONLY valid JSON for a realistic {meal} menu for "{name}" near OSU.
Format:
{{
  "provider": "{name}",
  "meal": "{meal}",
  "items": [
    {{
      "category": "Entrées",
      "options": [
        {{"name": "Item", "description": "Short desc", "price": 8.5}}
      ]
    }}
  ]
}}
Use 3–5 categories; each 3–5 options; prices as numbers.
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": "You output valid JSON only. No commentary."},
                {"role": "user", "content": prompt},
            ],
        )

        raw = (resp.choices[0].message.content or "").strip()

        # Strip code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)

        # Extract the largest JSON object if the model added stray text
        if not (raw.startswith("{") and raw.endswith("}")):
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            raw = m.group(0) if m else raw

        data = json.loads(raw)

        # ---- Normalize to guaranteed schema ----
        # If model returned a list, wrap as items
        if isinstance(data, list):
            data = {"items": data}

        # Force provider/meal
        data.setdefault("provider", name)
        data.setdefault("meal", meal)

        # Normalize items to a list of categories with options
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []

        # Case: model returned flat options instead of categories
        # Make a single category if needed.
        normalized_items = []
        for cat in items:
            if not isinstance(cat, dict):
                continue
            category = cat.get("category")
            options = cat.get("options")
            # If looks like an option row list, wrap it
            if isinstance(options, list):
                normalized_items.append({
                    "category": category or "Menu",
                    "options": [
                        {
                            "name": o.get("name", ""),
                            "description": o.get("description", ""),
                            "price": float(o.get("price", 0)) if isinstance(o.get("price", 0), (int, float, str)) else 0.0
                        }
                        for o in options if isinstance(o, dict)
                    ]
                })
            else:
                # If cat itself looks like an option, coerce
                if "name" in cat:
                    normalized_items.append({
                        "category": category or "Menu",
                        "options": [{
                            "name": cat.get("name", ""),
                            "description": cat.get("description", ""),
                            "price": float(cat.get("price", 0)) if isinstance(cat.get("price", 0), (int, float, str)) else 0.0
                        }]
                    })

        # If still empty, return a minimal placeholder
        if not normalized_items:
            normalized_items = [{
                "category": "Menu",
                "options": [
                    {"name": "Chef’s Special", "description": "House favorite.", "price": 9.5},
                    {"name": "Seasonal Bowl", "description": "Fresh and fast.", "price": 8.0},
                ],
            }]

        data["items"] = normalized_items
        return data

    except Exception as e:
        return skeleton(err=e)

# =====================================================
# Basic setup + helpers (unchanged)
# =====================================================
st.set_page_config(page_title="SyncBite –  AI-Powered Meal Planner", layout="wide")
st.title("SyncBite")

if "friends" not in st.session_state: st.session_state.friends = []
if "availability" not in st.session_state: st.session_state.availability = {}
if "restaurants_df" not in st.session_state: st.session_state.restaurants_df = None
if "user_latlon" not in st.session_state: st.session_state.user_latlon = None
if "global_wait_data" not in st.session_state: st.session_state.global_wait_data = {}

DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
OSU_OVAL = (40.0076,-83.0148)
OSU_BBOX = [40.0005,-83.0265,40.0188,-82.9965]
CROWD_MAP = {"Low":5,"Medium":15,"High":30}

def get_user_location():
    try:
        j = requests.get("https://ipinfo.io/json",timeout=5).json()
        if "loc" in j: return tuple(map(float,j["loc"].split(",")))
    except Exception: pass
    return OSU_OVAL

def haversine_km(a,b):
    R=6371; lat1,lon1=map(math.radians,a); lat2,lon2=map(math.radians,b)
    dlat,dlon=lat2-lat1,lon2-lon1
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def default_availability():
    return pd.DataFrame({"Day":DAYS,"Available":[False]*7,
                         "Start":[time(12,0)]*7,"End":[time(20,0)]*7})

def parse_opening_hours_basic(oh,idx):
    if not oh or not isinstance(oh,str): return []
    token=["Mo","Tu","We","Th","Fr","Sa","Su"][idx]
    wins=[]
    for rule in [r.strip() for r in oh.split(";") if r.strip()]:
        if "off" in rule.lower(): continue
        if not any(t in rule for t in [token,"Mo-Su"]): continue
        for m in re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})",rule):
            try:
                s=time(int(m[0].split(":")[0]),int(m[0].split(":")[1]))
                e=time(int(m[1].split(":")[0]),int(m[1].split(":")[1]))
                wins.append((s,e))
            except: continue
    return wins

def is_open_at(oh,idx,t):
    try:
        for s,e in parse_opening_hours_basic(oh,idx):
            if s<=t<e: return True
    except: pass
    return False

def estimate_wait(c): return CROWD_MAP.get(c,10)

def get_common_availability(avail_dict,sel):
    dfs=[avail_dict[f][avail_dict[f]["Available"]] for f in sel if f in avail_dict]
    if not dfs: return []
    merged=[]
    for d in DAYS:
        starts=[df[df["Day"]==d]["Start"].iloc[0] for df in dfs if d in df["Day"].values]
        ends=[df[df["Day"]==d]["End"].iloc[0] for df in dfs if d in df["Day"].values]
        if starts and ends:
            s=max(starts); e=min(ends)
            if s<e: merged.append((d,s,e))
    return merged

# =====================================================
# Tabs
# =====================================================
tab1, tab2, tab3 = st.tabs(["👥 Manage Friends", "🍴 Find OSU Restaurants", "🤖 Find Best Meal Spot + AI Menu"])

# =====================================================
# TAB 1: Manage Friends
# =====================================================
with tab1:
    st.header("Add / Remove Friends")
    c1,c2=st.columns([2,1])
    with c1:
        sel=st.selectbox("Select friend",["➕ Add new friend..."]+st.session_state.friends)
    with c2:
        if sel in st.session_state.friends and st.button("Remove"):
            st.session_state.friends.remove(sel)
            st.session_state.availability.pop(sel,None)
            st.rerun()
    if sel=="➕ Add new friend...":
        n=st.text_input("New friend name")
        if st.button("Add Friend"):
            if n and n not in st.session_state.friends:
                st.session_state.friends.append(n)
                st.session_state.availability[n]=default_availability()
                st.success(f"Added {n}")
                st.rerun()
            else: st.warning("Invalid or duplicate name.")

    st.divider()
    st.header("Manage Friends' Schedules")
    if not st.session_state.friends:
        st.info("No friends yet. Add them above first.")
    else:
        f=st.selectbox("Select friend to edit",st.session_state.friends)
        df=st.session_state.availability[f]
        nd=st.data_editor(df,use_container_width=True,hide_index=True,num_rows="fixed",
            column_config={"Day":st.column_config.Column(disabled=True),
                           "Available":st.column_config.CheckboxColumn(),
                           "Start":st.column_config.TimeColumn(format="HH:mm"),
                           "End":st.column_config.TimeColumn(format="HH:mm")})
        if st.button("Save Schedule"):
            st.session_state.availability[f]=nd; st.success("Saved!")

# =====================================================
# TAB 2: Find OSU Restaurants
# =====================================================
with tab2:
    st.header("Find OSU Restaurants")
    if not st.session_state.user_latlon:
        st.session_state.user_latlon=get_user_location()
    uloc=st.session_state.user_latlon
    st.caption(f"Using approximate location: {uloc[0]:.4f}, {uloc[1]:.4f}")

    def fetch_osu_restaurants():
        s,w,n,e=OSU_BBOX
        q=f"""[out:json][timeout:25];
        (node["amenity"~"restaurant|fast_food"]({s},{w},{n},{e});
         way["amenity"~"restaurant|fast_food"]({s},{w},{n},{e}););
        out center tags;"""
        d=requests.post("https://overpass-api.de/api/interpreter",data={"data":q},timeout=25).json()
        rows=[]
        for el in d.get("elements",[]):
            t=el.get("tags",{})
            name=t.get("name")
            if not name: continue
            lat=el.get("lat") or el.get("center",{}).get("lat")
            lon=el.get("lon") or el.get("center",{}).get("lon")
            if not lat or not lon: continue
            dist=haversine_km(uloc,(float(lat),float(lon)))
            rows.append({
                "Name":name,
                "Cuisine":t.get("cuisine",""),
                "Opening Hours":t.get("opening_hours",""),
                "Distance (km)":round(dist,2),
                "Your Rating":0.0,
                "Crowd Level":"Medium",
                "Expected Wait (min)":15
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["Name"]).sort_values("Distance (km)").reset_index(drop=True)

    if st.button("📍 Fetch OSU Restaurants"):
        with st.spinner("Fetching nearby restaurants..."):
            df=fetch_osu_restaurants()
            st.session_state.restaurants_df=df
            st.success(f"Loaded {len(df)} restaurants near campus.")

    if st.session_state.restaurants_df is not None:
        df = st.session_state.restaurants_df.copy()
        st.dataframe(df[["Name","Cuisine","Opening Hours","Distance (km)"]],
                     use_container_width=True, hide_index=True)

        st.subheader("Edit Restaurant Ratings / Crowd Levels / Wait Times")
        with st.form("edit_restaurant", clear_on_submit=False):
            rname = st.selectbox("Select restaurant", df["Name"].tolist())
            rating = st.slider("Your Rating (0–5)", 0.0, 5.0, float(df[df["Name"]==rname]["Your Rating"].iloc[0]), 0.1)
            crowd = st.selectbox("Crowd Level", ["Low","Medium","High"])
            wait = st.number_input("Expected Wait (min)", min_value=0, max_value=180, step=1, value=15)
            submitted = st.form_submit_button("Save")

        if submitted:
            df.loc[df["Name"]==rname, "Your Rating"] = rating
            wl = st.session_state.global_wait_data.get(rname, [])
            wl.append(estimate_wait(crowd)); wl.append(int(wait))
            wl = wl[-200:]
            st.session_state.global_wait_data[rname] = wl
            avg_wait = round(sum(wl)/len(wl), 1)
            df.loc[df["Name"]==rname, "Expected Wait (min)"] = avg_wait
            df.loc[df["Name"]==rname, "Crowd Level"] = crowd
            st.session_state.restaurants_df = df
            st.success(f"Updated {rname}. Avg wait now {avg_wait} min.")

# =====================================================
# TAB 3: Find Best Lunch Spot + Menu
# =====================================================
with tab3:
    st.header("Find Best Meal Spot and Menu Suggestion")
    if not st.session_state.friends or st.session_state.restaurants_df is None:
        st.info("Add friends and fetch restaurants first.")
    else:
        group=st.multiselect("Select friends for meal",st.session_state.friends,
                             default=st.session_state.friends[:2] if len(st.session_state.friends)>=2 else st.session_state.friends)
        if group:
            common=get_common_availability(st.session_state.availability,group)
            if not common:
                st.warning("No overlapping times found.")
            else:
                days=[c[0] for c in common]
                day=st.selectbox("Day",days)
                slot=next(s for s in common if s[0]==day)
                times=pd.date_range(datetime.combine(datetime.today(),slot[1]),
                                    datetime.combine(datetime.today(),slot[2]),freq="30min").time
                start=st.selectbox("Start time",times)

                w_rating=st.slider("Weight: Rating",0.0,1.0,0.5,0.05)
                w_wait=st.slider("Weight: Wait Time",0.0,1.0-w_rating,0.3,0.05)
                w_dist=1.0-w_rating-w_wait

                if st.button("🔍 Find the Best Place"):
                    df=st.session_state.restaurants_df.copy()
                    idx=DAYS.index(day)
                    df["Is Open"]=df.apply(lambda r:is_open_at(r["Opening Hours"],idx,start),axis=1)
                    df=df[df["Is Open"]==True]
                    if df.empty:
                        st.warning("No open restaurants at that time.")
                    else:
                        def normalize(series):
                            if series.max() == series.min(): return np.random.uniform(0.4,0.6,len(series))
                            return (series-series.min())/(series.max()-series.min())

                        df["Your Rating"] = df["Your Rating"].replace(0,2.5)
                        df["Expected Wait (min)"] = df["Expected Wait (min)"].replace(0,15)

                        norm_rating = normalize(df["Your Rating"])
                        norm_wait = normalize(df["Expected Wait (min)"])
                        norm_dist = normalize(df["Distance (km)"])

                        noise = np.random.normal(0, 0.0001, len(df))
                        df["Score"] = (w_rating*norm_rating)+(w_wait*(1-norm_wait))+(w_dist*(1-norm_dist))+noise

                        df=df.sort_values("Score",ascending=False)
                        top=df.head(5)
                        st.dataframe(top[["Name","Cuisine","Distance (km)","Your Rating","Expected Wait (min)","Score"]],
                                     use_container_width=True)
                        best=top.iloc[0]
                        st.success(f"🏆 Best spot for {', '.join(group)}: {best['Name']} ({best['Distance (km)']:.2f} km away)")

                        st.subheader("Menu Suggestion (AI-generated)")
                        with st.spinner("Generating menu..."):
                            menu=fetch_menu(best["Name"],"lunch" if 10<=start.hour<16 else "dinner",datetime.now().date())

                        menu = fetch_menu(best["Name"], "lunch" if 10 <= start.hour < 16 else "dinner", datetime.now().date())

# 🔍 Debug line
                        if "error" in menu:
                            st.error(f"OpenAI menu generation failed: {menu['error']}")

                        st.write(f"**Provider:** {menu.get('provider',best['Name'])} | **Meal:** {menu.get('meal','N/A')}")
                        st.write("---")
                        for cat in menu.get("items", []):
                            st.markdown(f"#### {cat.get('category', 'Menu')}")
                            for opt in cat.get("options", []):
                                st.markdown(f"- **{opt.get('name','')}** — {opt.get('description','')} ${opt.get('price','')}")
