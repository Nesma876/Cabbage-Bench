# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CABBAGE-Bench — TOUT EN UNE CELLULE                                      ║
# ║  Colle ce bloc dans UNE cellule Kaggle et execute.                        ║
# ║                                                                           ║
# ║  Sans Internet : tester_synthetique() et tester_lignees() tournent.        ║
# ║  Avec Internet (Settings > Internet > On) :                               ║
# ║      especes = explorer(seuil=300)                                        ║
# ║      d = construire("Klebsiella pneumoniae", seuil=300)                   ║
# ║      d = filtrer(d)                                                       ║
# ║      cache = telecharger_genes(d["meta"], "genes_kp.pkl", limite=20)      ║
# ║      cache = telecharger_genes(d["meta"], "genes_kp.pkl")   # le reste    ║
# ║      lignees = construire_lignees(cache, genes_amr=d["genes"])            ║
# ║      d["splits"]["lignee"] = splits_par_lignee(d["meta"], lignees)        ║
# ║      controle_fuite(d["meta"], lignees, d["splits"])                      ║
# ║      res = evaluer(d)                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import subprocess, sys
try:
    import duckdb
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=False)

import gzip, io, os, pickle, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

FTP = "https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases"
FTP_G = "https://ftp.ebi.ac.uk/pub/databases/amr_portal/genomes"
RELEASE = "2025-11"

def _con():
    """Connexion DuckDB, avec le proxy du systeme si la machine en impose un.

    Sur un calculateur comme Jean Zay, l'acces exterieur passe par un proxy
    declare dans les variables d'environnement. urllib les lit tout seul,
    DuckDB non : il faut le lui dire.
    """
    import duckdb, os, re
    c = duckdb.connect()
    try:
        c.execute("LOAD httpfs;")               # deja installee ?
    except Exception:                           # noqa: BLE001
        try:
            c.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception as e:                  # noqa: BLE001
            raise RuntimeError(
                "DuckDB ne peut pas installer l'extension httpfs, qui se "
                "telecharge depuis extensions.duckdb.org.\n"
                "Si ce domaine est bloque :\n"
                "  1. recupere le fichier sur une machine ayant Internet,\n"
                "  2. copie-le ici,\n"
                "  3. remplace l'installation par :\n"
                "       INSTALL '/chemin/httpfs.duckdb_extension';\n"
                f"Erreur d'origine : {e}") from e
    proxy = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
    if proxy:
        p = re.sub(r"^https?://", "", proxy).rstrip("/")
        hote, _, port = p.partition(":")
        try:
            c.execute(f"SET http_proxy='{hote}:{port or 3128}'")
            print(f"  (DuckDB passe par le proxy {hote}:{port or 3128})")
        except Exception as e:                      # noqa: BLE001
            print(f"  (proxy non applique a DuckDB : {type(e).__name__})")
    return c

# ═══════════════════════ 1. EXPLORATION ═══════════════════════════════════
def explorer(release=RELEASE, seuil=300,
             especes=("Escherichia coli", "Klebsiella pneumoniae")):
    url, con = f"{FTP}/{release}/phenotype.parquet", _con()
    print(f"Lecture : {url}\n=== Volume total ===")
    print(con.execute(f"""SELECT count(*) entrees, count(DISTINCT BioSample_ID) isolats,
          count(DISTINCT antibiotic_name) antibiotiques FROM read_parquet('{url}')
          """).df().to_string(index=False))
    pe = con.execute(f"""SELECT species, count(DISTINCT BioSample_ID) tous,
          count(DISTINCT CASE WHEN assembly_ID IS NOT NULL AND assembly_ID <> ''
                THEN BioSample_ID END) avec_assemblage,
          count(DISTINCT antibiotic_name) antibiotiques,
          count(DISTINCT ISO_country_code) pays
          FROM read_parquet('{url}') GROUP BY species ORDER BY tous DESC LIMIT 25""").df()
    pe["taux_assemblage"] = (100 * pe.avec_assemblage / pe.tous).round(1)
    print("\n=== Par espece ==="); print(pe.to_string(index=False))
    for esp in especes:
        df = con.execute(f"""SELECT antibiotic_name,
              count(DISTINCT BioSample_ID) isolats_testes FROM read_parquet('{url}')
              WHERE species = ? AND assembly_ID IS NOT NULL AND assembly_ID <> ''
              AND coalesce(Updated_phenotype_EUCAST, resistance_phenotype) IS NOT NULL
              GROUP BY antibiotic_name HAVING count(DISTINCT BioSample_ID) >= {seuil}
              ORDER BY isolats_testes DESC""", [esp]).df()
        print(f"\n=== {esp} : antibiotiques >= {seuil} isolats ===")
        print(df.to_string(index=False) if not df.empty else "  (aucun)")
    return pe

# ═══════════════════════ 2. MATRICES MULTI-LABEL ══════════════════════════
def _splits(meta, graine=0):
    """Trois decoupages. Les valeurs manquantes vont toujours en entrainement :
    on ne teste jamais sur un isolat dont on ignore la region ou l'annee."""
    rng = np.random.default_rng(graine); t = rng.random(len(meta))
    out = pd.DataFrame(index=meta.index)
    out["aleatoire"] = np.where(t < .7, "train", np.where(t < .85, "val", "test"))

    reg = meta["region"].astype("object").where(meta["region"].notna(), None)
    c = reg.dropna().value_counts()
    if len(c) >= 2:
        cible = c.index[1]          # 2e region la mieux fournie -> test
        out["region"] = np.where((reg == cible).fillna(False), "test", "train")
        print(f"  decoupage region : test = {cible} ({int(c.iloc[1])} isolats)")
    else:
        out["region"] = "train"

    # Coupure temporelle : on cherche l'annee qui met environ 20 % en test.
    # Un simple quantile peut donner un test vide quand beaucoup d'isolats
    # partagent la derniere annee.
    a = pd.to_numeric(meta["annee"], errors="coerce").astype("float64")
    connues = a.dropna()
    coupe = None
    if len(connues):
        for an in sorted(connues.unique(), reverse=True):
            if (connues >= an).mean() >= 0.15:
                coupe = an
                break
    en_test = (a.notna() & (a >= coupe)).to_numpy() if coupe is not None else None
    if en_test is not None and en_test.mean() <= 0.5:
        out["periode"] = np.where(en_test, "test", "train")
        print(f"  decoupage periode : test = annees >= {coupe:.0f} "
              f"({int(en_test.sum())} isolats ; "
              f"{int(a.isna().sum())} annees manquantes -> train)")
    else:
        # une seule annee, ou coupure qui bascule tout : le decoupage n'a pas de sens
        out["periode"] = "train"
        print("  decoupage periode : inutilisable (annees trop peu variees) "
              "-> colonne a ignorer")
    return out

def construire(espece="Escherichia coli", release=RELEASE, seuil=300):
    """Phenotype EUCAST 2025 d'abord, puis CLSI, puis le phenotype d'origine.

    Un meme couple isolat-antibiotique peut apparaitre plusieurs fois avec des
    resultats differents. On agrege par la part de tests resistants : les cas
    contradictoires (part strictement entre 0 et 1) sont ecartes plutot que
    tranches au hasard, et leur nombre est affiche.
    """
    con, url = _con(), f"{FTP}/{release}/phenotype.parquet"
    df = con.execute(f"""
        WITH lignes AS (
            SELECT BioSample_ID, assembly_ID, ISO_country_code pays,
                   geographical_region region, collection_year annee, host,
                   isolation_source_category src_cat, antibiotic_name,
                   -- transtypage explicite : si une espece n'a aucune valeur
                   -- dans une colonne, DuckDB lui prete un type numerique et
                   -- coalesce echoue
                   CASE WHEN lower(coalesce(CAST(Updated_phenotype_EUCAST AS VARCHAR),
                            CAST(Updated_phenotype_CLSI AS VARCHAR),
                            CAST(resistance_phenotype AS VARCHAR)))
                             LIKE 'resistant%' THEN 1
                        WHEN lower(coalesce(CAST(Updated_phenotype_EUCAST AS VARCHAR),
                            CAST(Updated_phenotype_CLSI AS VARCHAR),
                            CAST(resistance_phenotype AS VARCHAR)))
                             LIKE 'susceptible%' THEN 0 END y
            FROM read_parquet('{url}')
            WHERE species = ? AND assembly_ID IS NOT NULL AND assembly_ID <> ''),
        agg AS (
            SELECT BioSample_ID, any_value(assembly_ID) assembly_ID,
                   any_value(pays) pays, any_value(region) region,
                   any_value(annee) annee, any_value(host) host,
                   any_value(src_cat) src_cat, antibiotic_name,
                   avg(y) part_r
            FROM lignes WHERE y IS NOT NULL
            GROUP BY BioSample_ID, antibiotic_name)
        SELECT *, CAST(part_r AS INTEGER) y FROM agg WHERE part_r IN (0, 1)""",
        [espece]).df()
    n_contr = con.execute(f"""
        WITH lignes AS (
            SELECT BioSample_ID, antibiotic_name,
                   -- transtypage explicite : si une espece n'a aucune valeur
                   -- dans une colonne, DuckDB lui prete un type numerique et
                   -- coalesce echoue
                   CASE WHEN lower(coalesce(CAST(Updated_phenotype_EUCAST AS VARCHAR),
                            CAST(Updated_phenotype_CLSI AS VARCHAR),
                            CAST(resistance_phenotype AS VARCHAR)))
                             LIKE 'resistant%' THEN 1
                        WHEN lower(coalesce(CAST(Updated_phenotype_EUCAST AS VARCHAR),
                            CAST(Updated_phenotype_CLSI AS VARCHAR),
                            CAST(resistance_phenotype AS VARCHAR)))
                             LIKE 'susceptible%' THEN 0 END y
            FROM read_parquet('{url}')
            WHERE species = ? AND assembly_ID IS NOT NULL AND assembly_ID <> '')
        SELECT sum(CASE WHEN part_r NOT IN (0,1) THEN 1 ELSE 0 END) c
        FROM (SELECT BioSample_ID, antibiotic_name, avg(y) part_r FROM lignes
              WHERE y IS NOT NULL GROUP BY BioSample_ID, antibiotic_name)""",
        [espece]).fetchone()[0] or 0
    if n_contr:
        print(f"{int(n_contr)} couples isolat-antibiotique contradictoires ecartes")
    cnt = df.groupby("antibiotic_name")["BioSample_ID"].nunique()
    antibios = sorted(cnt[cnt >= seuil].index)
    df = df[df["antibiotic_name"].isin(antibios)]
    isolats = sorted(df["BioSample_ID"].unique())
    ii = {b: i for i, b in enumerate(isolats)}; aa = {a: j for j, a in enumerate(antibios)}
    Y = np.zeros((len(isolats), len(antibios)), np.int8); M = np.zeros_like(Y)
    Y[df["BioSample_ID"].map(ii), df["antibiotic_name"].map(aa)] = df["y"].to_numpy()
    M[df["BioSample_ID"].map(ii), df["antibiotic_name"].map(aa)] = 1
    print(f"{espece} : {Y.shape[0]} isolats x {Y.shape[1]} antibiotiques, "
          f"{100*M.mean():.1f} % de cases testees")
    meta = (df.groupby("BioSample_ID").agg(assembly_ID=("assembly_ID","first"),
            pays=("pays","first"), region=("region","first"), annee=("annee","first"),
            host=("host","first"), src_cat=("src_cat","first")).reindex(isolats))
    con.register("vus", pd.DataFrame({"BioSample_ID": isolats}))
    g = con.execute(f"""SELECT DISTINCT g.BioSample_ID, g.amr_element_symbol gene
        FROM read_parquet('{FTP}/{release}/genotype.parquet') g
        JOIN vus v ON v.BioSample_ID = g.BioSample_ID
        WHERE g.species = ? AND g.amr_element_symbol IS NOT NULL""", [espece]).df()
    genes = sorted(g["gene"].unique())
    X = sparse.csr_matrix((np.ones(len(g), np.int8),
        (g["BioSample_ID"].map(ii).to_numpy(),
         g["gene"].map({v: i for i, v in enumerate(genes)}).to_numpy())),
        shape=(len(isolats), len(genes)))
    print(f"{len(genes)} genes de resistance distincts")
    return {"X": X, "Y": Y, "M": M, "meta": meta, "splits": _splits(meta),
            "antibiotiques": antibios, "genes": genes}

# ═══════════════════════ 3. FILTRAGE DES ANTIBIOTIQUES ════════════════════
# Paires utilisees pour DETECTER une enzyme (test de confirmation BLSE),
# pas pour traiter. Les vraies associations (pip-tazo, amox-clav, amp-sulbactam,
# ceftazidime-avibactam) sont bien des traitements et restent incluses.
TESTS_DIAGNOSTIC = {"ceftazidime-clavulanic acid", "cefotaxime-clavulanic acid",
    "cefepime-clavulanic acid", "cefpodoxime-clavulanic acid",
    "ceftriaxone-clavulanic acid"}
VETERINAIRES = {"enrofloxacin", "ceftiofur", "florfenicol", "tilmicosin",
    "tulathromycin", "tylosin", "danofloxacin", "marbofloxacin", "cefquinome",
    "virginiamycin", "carbadox"}
A_VERIFIER = {"pentizidone", "fludalanine"}   # MK642/MK641, experimentaux (1984)

def filtrer(d, veterinaire=True, a_verifier=True, contexte=None):
    raisons = {}
    for a in d["antibiotiques"]:
        c = a.strip().lower()
        if c in TESTS_DIAGNOSTIC: raisons[a] = "test de confirmation BLSE"
        elif veterinaire and c in VETERINAIRES: raisons[a] = "antibiotique veterinaire"
        elif a_verifier and c in A_VERIFIER: raisons[a] = "compose experimental"
    if raisons:
        print("Antibiotiques exclus :")
        for a, r in sorted(raisons.items()): print(f"  - {a:32} {r}")
    else: print("Aucun antibiotique a exclure.")
    g = [j for j, a in enumerate(d["antibiotiques"]) if a not in raisons]
    o = dict(d); o["antibiotiques"] = [d["antibiotiques"][j] for j in g]
    o["Y"], o["M"], o["exclus"] = d["Y"][:, g], d["M"][:, g], raisons
    if contexte and "host" in o["meta"].columns:
        h = o["meta"]["host"].fillna("").str.lower()
        hum = h.str.contains("homo sapiens|human", regex=True)
        sel = (hum if contexte == "humain" else ~hum).to_numpy()
        print(f"\nContexte « {contexte} » : {sel.sum()} isolats sur {len(sel)}")
        o["Y"], o["M"], o["X"] = o["Y"][sel], o["M"][sel], o["X"][sel]
        o["meta"], o["splits"] = o["meta"][sel], o["splits"][sel]
    print(f"\nApres filtrage : {o['Y'].shape[0]} isolats x {o['Y'].shape[1]} "
          f"antibiotiques, {100*o['M'].mean():.1f} % de cases testees")
    return o

# ═══════════════════════ 4. LIGNEES GENETIQUES ════════════════════════════
def chemin_gff(acc):
    acc = str(acc).strip()
    if acc[:4] in ("GCA_", "GCF_"): pre, ch = acc[:3], acc[4:].split(".")[0]
    elif acc.startswith("ERZ"):     pre, ch = "ERZ", acc[3:]
    else: return None
    ch = ch.zfill(6)
    return f"{FTP_G}/{pre}/{ch[:3]}/{ch[3:6]}/{acc}/{acc}_annotations.gff.gz"

def _genes_gff(txt):
    out = set()
    for l in io.StringIO(txt):
        if not l or l[0] == "#": continue
        c = l.rstrip("\n").split("\t")
        if len(c) < 9 or c[2] not in ("CDS", "gene"): continue
        for a in c[8].split(";"):
            if a.startswith("Name="): out.add(a[5:]); break
            if a.startswith("gene="): out.add(a[5:]); break
    return out

def _un(acc, delai=30, essais=3):
    """Un 404 est definitif : le genome n'est pas dans le portail, on n'insiste pas.
    Une erreur reseau est passagere : on reessaie, en doublant l'attente."""
    u = chemin_gff(acc)
    if u is None: return acc, None, "accession non reconnue"
    derniere = "inconnu"
    for i in range(essais):
        try:
            with urllib.request.urlopen(u, timeout=delai) as r: brut = r.read()
            return acc, _genes_gff(gzip.decompress(brut).decode("utf-8", "replace")), None
        except urllib.error.HTTPError as e:
            if e.code == 404: return acc, None, "HTTP 404"
            derniere = f"HTTP {e.code}"
        except Exception as e:
            derniere = type(e).__name__
        if i + 1 < essais: time.sleep(2 ** i)
    return acc, None, derniere

def telecharger_genes(meta, fichier, limite=None, fils=8):
    """Reprenable : relancer reprend ou la fonction s'est arretee."""
    deja = {}
    if os.path.exists(fichier):
        deja = pickle.load(open(fichier, "rb")); print(f"{len(deja)} deja en cache")
    ids = [a for a in meta["assembly_ID"].dropna().unique() if a not in deja]
    if limite: ids = ids[:limite]
    if not ids: print("Rien a telecharger."); return deja
    print(f"Telechargement de {len(ids)} annotations, {fils} en parallele...")
    echecs, faits = {}, 0
    with ThreadPoolExecutor(max_workers=fils) as ex:
        for fut in as_completed({ex.submit(_un, a): a for a in ids}):
            acc, g, err = fut.result()
            if g is None: echecs[acc] = err
            else: deja[acc] = g
            faits += 1
            if faits % 100 == 0:
                print(f"  {faits}/{len(ids)} ({len(echecs)} echecs)")
                pickle.dump(deja, open(fichier, "wb"))
    pickle.dump(deja, open(fichier, "wb"))
    print(f"Termine : {len(deja)} en cache, {len(echecs)} echecs")
    if echecs:
        print(pd.Series(list(echecs.values())).value_counts().to_string())
        k = list(echecs)[0]; print("Exemple :", k, "->", chemin_gff(k))
    return deja


def _lignees_par_echantillon(X, accs, ech, seuil, max_part, min_couv, graine):
    """Pour les tres grandes especes : regrouper un echantillon, rattacher le reste.

    Le regroupement hierarchique exige une matrice carree, impraticable au-dela
    d'environ 20 000 genomes. On tire donc un echantillon, on le regroupe
    exactement, puis chaque genome restant rejoint la lignee dont le centre est
    le plus proche au sens de Jaccard.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(graine)
    n = X.shape[0]
    pris = np.sort(rng.choice(n, min(ech, n), replace=False))
    Xe = X[pris].toarray().astype(bool)
    D = pairwise_distances(Xe, metric="jaccard", n_jobs=-1)
    Z = linkage(squareform(D, checks=False), method="complete"); del D

    lab_e, retenu = None, seuil
    if seuil is not None:
        lab_e = fcluster(Z, t=seuil, criterion="distance")
    else:
        for s_essai in np.arange(0.20, 0.70, 0.025):
            essai = fcluster(Z, t=s_essai, criterion="distance")
            t = pd.Series(essai).value_counts()
            if (100*t.iloc[0]/len(pris) <= max_part
                    and 100*t[t >= 5].sum()/len(pris) >= min_couv):
                lab_e, retenu = essai, s_essai; break
        if lab_e is None:
            retenu = 0.45; lab_e = fcluster(Z, t=retenu, criterion="distance")
    print(f"  seuil calibre sur l'echantillon : {retenu:.3f}, "
          f"{len(set(lab_e))} lignees")

    # centre de chaque lignee = frequence de chaque gene dans la lignee
    etiquettes = np.unique(lab_e)
    C = np.vstack([Xe[lab_e == l].mean(axis=0) for l in etiquettes])
    Cb = C > 0.5                                  # genes majoritaires
    lab = np.empty(n, dtype=int)
    lab[pris] = lab_e
    reste = np.setdiff1d(np.arange(n), pris)
    for debut in range(0, len(reste), 2000):      # par tranches, pour la memoire
        bloc = reste[debut:debut+2000]
        d = pairwise_distances(X[bloc].toarray().astype(bool), Cb,
                               metric="jaccard", n_jobs=-1)
        lab[bloc] = etiquettes[d.argmin(axis=1)]
    print(f"  {len(reste)} genomes rattaches a leur lignee la plus proche")

    s = pd.Series(lab, index=accs, name="lignee"); t = s.value_counts()
    print(f"{s.nunique()} lignees ; la plus grande = {t.iloc[0]} genomes "
          f"({100*t.iloc[0]/n:.1f} %), mediane {int(t.median())}, "
          f"{100*t[t>=5].sum()/n:.1f} % des genomes dans un groupe d'au moins 5")
    return s


def construire_lignees(src, genes_amr=(), seuil=None, bande=(0.01, 0.99),
                       max_part_lignee=15, min_couverture=95,
                       max_go=8, ech=12000, graine=0):
    """Regroupe les genomes par distance de Jaccard sur le contenu en genes.

    Deux precautions, apprises sur les donnees reelles :

    1. On ne garde que le genome ACCESSOIRE, c'est-a-dire les genes presents
       dans 1 a 99 % des genomes. Les genes du coeur, presents partout, ne
       distinguent rien ; les genes vus une ou deux fois sont surtout du bruit
       d'annotation.

    2. Le seuil est CALIBRE automatiquement, pas fixe d'avance. Chez
       K. pneumoniae la distance mediane entre deux genomes vaut 0,41 : un
       seuil de 0,05 laisserait 76 % des genomes isoles, et le decoupage
       "par lignee" reviendrait alors a un tirage aleatoire. On choisit donc
       le seuil le plus strict tel qu'aucune lignee ne depasse `max_part_lignee` %
       des genomes et qu'au moins `min_couverture` % des genomes aient au
       moins quatre proches dans leur groupe.
       Passer `seuil=0.3` force une valeur si on veut la fixer soi-meme.

    Les genes de resistance sont EXCLUS : sans cela on grouperait les genomes
    selon ce qu'on cherche a predire, recreant la fuite qu'on veut supprimer.
    """
    cache = pickle.load(open(src, "rb")) if isinstance(src, str) else src
    accs = sorted(cache); exc = {g.lower() for g in genes_amr}
    n = len(accs)
    cnt = {}
    for a in accs:
        for g in cache[a]:
            if g.lower() not in exc: cnt[g] = cnt.get(g, 0) + 1
    lo, hi = bande[0] * n, bande[1] * n
    vocab = sorted(g for g, c in cnt.items() if lo <= c < hi)
    idx = {g: j for j, g in enumerate(vocab)}
    print(f"{n} assemblages, {len(vocab)} genes accessoires "
          f"({len(exc)} genes de resistance exclus)")
    li, lj = [], []
    for i, a in enumerate(accs):
        for g in cache[a]:
            j = idx.get(g)
            if j is not None: li.append(i); lj.append(j)
    X = sparse.csr_matrix((np.ones(len(li), np.int8), (li, lj)),
                          shape=(len(accs), len(vocab)))
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import pairwise_distances
    # La matrice de distances est carree : 29 000 genomes demandent environ
    # 10 Go au pic. Au-dela du plafond, on regroupe sur un echantillon puis on
    # rattache les autres genomes a leur lignee la plus proche.
    go = n * n * 8 / 1e9 * 1.5
    if go > max_go:
        print(f"  {n} genomes demanderaient {go:.1f} Go : regroupement sur un "
              f"echantillon de {ech} genomes, puis rattachement des autres")
        return _lignees_par_echantillon(X, accs, ech, seuil, max_part_lignee,
                                        min_couverture, graine)
    print(f"  calcul des distances de Jaccard ({go:.1f} Go au pic)...")
    D = pairwise_distances(X.toarray().astype(bool), metric="jaccard", n_jobs=-1)
    print(f"  distance mediane entre genomes : "
          f"{np.median(D[np.triu_indices(n, 1)]):.3f}")
    # Lien COMPLET et non moyen : sur donnees reelles, le lien moyen enchaine
    # les genomes de proche en proche et produit un bloc geant entoure de
    # singletons. Chez K. pneumoniae, le lien moyen donnait une lignee de 21 %
    # des genomes avec une mediane de 5, contre 8 % et une mediane de 11 pour
    # le lien complet, a couverture egale.
    Z = linkage(squareform(D, checks=False), method="complete")
    del D

    if seuil is not None:
        lab, retenu = fcluster(Z, t=seuil, criterion="distance"), seuil
    else:
        lab, retenu = None, None
        for s_essai in np.arange(0.20, 0.70, 0.025):
            essai = fcluster(Z, t=s_essai, criterion="distance")
            t = pd.Series(essai).value_counts()
            pct_max = 100 * t.iloc[0] / n
            # part des genomes ayant au moins quatre proches : c'est elle qui
            # dit si le decoupage protege vraiment, pas le nombre de groupes
            pct_utiles = 100 * t[t >= 5].sum() / n
            if pct_max <= max_part_lignee and pct_utiles >= min_couverture:
                lab, retenu = essai, s_essai
                break
        if lab is None:
            retenu = 0.45
            lab = fcluster(Z, t=retenu, criterion="distance")
            print(f"  ATTENTION : aucun seuil ne tient les criteres ; repli sur "
                  f"{retenu}. Verifie la distribution des tailles ci-dessous.")
        else:
            print(f"  seuil calibre : {retenu:.3f}")

    s = pd.Series(lab, index=accs, name="lignee"); t = s.value_counts()
    print(f"{s.nunique()} lignees ; la plus grande = {t.iloc[0]} genomes "
          f"({100*t.iloc[0]/n:.1f} %), mediane {int(t.median())}, "
          f"{100*t[t>=5].sum()/n:.1f} % des genomes dans un groupe d'au moins 5")
    return s

def splits_par_lignee(meta, lignees, part_test=0.3, graine=0):
    rng = np.random.default_rng(graine); lig = meta["assembly_ID"].map(lignees)
    if lig.isna().sum(): print(f"{lig.isna().sum()} isolats sans lignee -> train")
    t = lig.value_counts(); cible, cum, test = part_test * len(meta), 0, set()
    for l in rng.permutation(t.index.to_numpy()):
        if cum >= cible: break
        test.add(l); cum += t[l]
    out = pd.Series(np.where(lig.isin(test), "test", "train"), index=meta.index)
    print(f"Lignee : {(out=='test').sum()} isolats en test "
          f"({100*(out=='test').mean():.0f} %), {len(test)}/{lig.nunique()} lignees")
    return out

def controle_fuite(meta, lignees, splits, col="lignee"):
    lig = meta["assembly_ID"].map(lignees)
    com = (set(lig[splits[col] == "train"].dropna())
           & set(lig[splits[col] == "test"].dropna()))
    print(f"ATTENTION : {len(com)} lignees des deux cotes" if com
          else "Aucune fuite : lignees de test absentes de l'entrainement")
    return com

# ═══════════════════════ 5. MODELES ET METRIQUES ══════════════════════════
def _ece(y, p, bins=10):
    b = np.linspace(0, 1, bins + 1); tot = 0.
    for lo, hi in zip(b[:-1], b[1:]):
        s = (p > lo) & (p <= hi)
        if s.sum(): tot += s.mean() * abs(y[s].mean() - p[s].mean())
    return tot

def _metriques(y, p, seuil=.5):
    if len(np.unique(y)) < 2: return None
    pr = (p >= seuil).astype(int); ns, nr = (y == 0).sum(), (y == 1).sum()
    return {"n": int(len(y)), "pct_resistant": round(100*nr/len(y), 1),
            "auroc": round(roc_auc_score(y, p), 3),
            "auprc": round(average_precision_score(y, p), 3),
            "f1": round(f1_score(y, pr, zero_division=0), 3),
            # majeure : sensible annonce resistant ; tres majeure : l'inverse (grave)
            "err_majeure_pct": round(100*((pr==1)&(y==0)).sum()/max(ns,1), 1),
            "err_tres_majeure_pct": round(100*((pr==0)&(y==1)).sum()/max(nr,1), 1),
            "ece": round(_ece(y, p), 3)}

def _separe(X, Y, M, tr):
    P = np.full(Y.shape, .5)
    for j in range(Y.shape[1]):
        s = tr & (M[:, j] == 1)
        if s.sum() < 20 or len(np.unique(Y[s, j])) < 2: continue
        clf = LogisticRegression(max_iter=2000); clf.fit(X[s], Y[s, j])
        P[:, j] = clf.predict_proba(X)[:, 1]
    return P

def _multi(X, Y, M, tr, h=32, epochs=2000, lr=.03, l2=1e-4, graine=0):
    """Goulot partage X -> h -> k, perte BCE masquee.
    Sans couche cachee, une sortie multi-sigmoide equivaut mathematiquement
    a des modeles independants : le goulot est ce qui fait le partage."""
    rng = np.random.default_rng(graine)
    Xd = np.asarray(X.todense(), float); d, k = Xd.shape[1], Y.shape[1]
    W1 = rng.normal(0, .1, (d, h)); b1v = np.zeros(h)
    W2 = rng.normal(0, .1, (h, k)); b2v = np.zeros(k)
    st = {n: [np.zeros_like(p), np.zeros_like(p)]
          for n, p in [("W1", W1), ("b1", b1v), ("W2", W2), ("b2", b2v)]}
    be1, be2, eps = .9, .999, 1e-8
    Xtr, Ytr, Mtr = Xd[tr], Y[tr].astype(float), M[tr].astype(float)
    ncol = np.maximum(Mtr.sum(0), 1.)
    def adam(nom, p, g, t):
        m, v = st[nom]
        m[:] = be1*m + (1-be1)*g; v[:] = be2*v + (1-be2)*g**2
        p -= lr * (m/(1-be1**t)) / (np.sqrt(v/(1-be2**t)) + eps)
    for t in range(1, epochs+1):
        a1 = np.tanh(Xtr @ W1 + b1v)
        p = 1/(1+np.exp(-(a1 @ W2 + b2v)))
        g2 = (p - Ytr) * Mtr / ncol          # le masque annule les non testes
        g1 = (g2 @ W2.T) * (1 - a1**2)
        adam("W2", W2, a1.T @ g2 + l2*W2, t); adam("b2", b2v, g2.sum(0), t)
        adam("W1", W1, Xtr.T @ g1 + l2*W1, t); adam("b1", b1v, g1.sum(0), t)
    return 1/(1+np.exp(-(np.tanh(Xd @ W1 + b1v) @ W2 + b2v)))

def evaluer(d):
    X, Y, M = d["X"], d["Y"], d["M"]; lignes = []
    for dec in d["splits"].columns:
        tr = (d["splits"][dec] == "train").to_numpy()
        te = (d["splits"][dec] == "test").to_numpy()
        if not tr.sum() or not te.sum(): continue
        for nom, P in [("separe", _separe(X, Y, M, tr)), ("multi_label", _multi(X, Y, M, tr))]:
            for j, ab in enumerate(d["antibiotiques"]):
                s = te & (M[:, j] == 1)
                if s.sum() < 20: continue
                m = _metriques(Y[s, j], P[s, j])
                if m: lignes.append({"decoupage": dec, "modele": nom, "antibiotique": ab, **m})
    res = pd.DataFrame(lignes)
    if not res.empty:
        print("\n=== AUROC moyen ===")
        print(res.pivot_table(index="decoupage", columns="modele", values="auroc",
                              aggfunc="mean").round(3).to_string())
        print("\n=== Erreur tres majeure moyenne (%) ===")
        print(res.pivot_table(index="decoupage", columns="modele",
              values="err_tres_majeure_pct", aggfunc="mean").round(1).to_string())
    return res

# ═══════════════════════ 6. TESTS SANS RESEAU ═════════════════════════════
def tester_synthetique(n=800, d=60, k=8, graine=0):
    rng = np.random.default_rng(graine)
    regions = rng.choice(["Europe", "Americas", "Asia"], n, p=[.5, .3, .2])
    annees = rng.integers(2005, 2023, n)
    X = (rng.random((n, d)) < .08).astype(np.int8); W = np.zeros((d, k))
    for j in range(k):
        for g in rng.choice(d, 2, replace=False): W[g, j] = 3.
    Y = ((1/(1+np.exp(-(X@W - 1.5 + rng.normal(0, .5, (n, k)))))) > .5).astype(np.int8)
    M = (rng.random((n, k)) < .5).astype(np.int8)
    M[np.arange(n), rng.integers(0, k, n)] = 1
    iso = [f"SAMTEST{i:05d}" for i in range(n)]
    meta = pd.DataFrame({"assembly_ID": [f"GCA_{i:09d}.1" for i in range(n)],
        "pays": rng.choice(["FRA","USA","CHN","DZA"], n), "region": regions,
        "annee": annees, "host": "Homo sapiens"}, index=iso)
    meta.index.name = "BioSample_ID"
    print(f"Jeu synthetique : {n} isolats, {k} antibiotiques, {100*M.mean():.0f} % testees")
    return evaluer({"X": sparse.csr_matrix(X), "Y": Y, "M": M, "meta": meta,
        "splits": _splits(meta), "genes": [f"gene_{g}" for g in range(d)],
        "antibiotiques": [f"antibio_{j}" for j in range(k)]})

def tester_lignees():
    """Verifie le regroupement et l'absence de fuite, sans reseau."""
    rng = np.random.default_rng(0); cache, verite = {}, {}
    for lig in range(12):
        noyau = {f"gene_{g}" for g in rng.choice(400, 150, replace=False)}
        for m in range(20):
            acc = f"GCA_{lig:03d}{m:03d}.1"
            g = set(noyau) - set(rng.choice(sorted(noyau), 10, replace=False))
            g |= {f"gene_{x}" for x in rng.choice(400, 10, replace=False)}
            g |= {"blaTEM-1", "sul1"}
            cache[acc] = g; verite[acc] = lig
    s = construire_lignees(cache, genes_amr=["blaTEM-1", "sul1"], seuil=0.35)
    from sklearn.metrics import adjusted_rand_score
    print("Indice de Rand ajuste :",
          round(adjusted_rand_score([verite[a] for a in s.index], s.to_numpy()), 3))
    meta = pd.DataFrame({"assembly_ID": s.index}, index=s.index)
    sp = pd.DataFrame({"lignee": splits_par_lignee(meta, s)})
    controle_fuite(meta, s, sp)
    print("Chemin FTP type :", chemin_gff("GCA_000194885.2"))
    return s

print("Pret.")
print("  explorer() | construire() | filtrer() | evaluer()")
print("  telecharger_genes() | construire_lignees() | splits_par_lignee() | controle_fuite()")
print("  tester_synthetique() | tester_lignees()\n")
tester_synthetique()
print()
tester_lignees()
