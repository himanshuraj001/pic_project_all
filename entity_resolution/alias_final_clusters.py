import pandas as pd
import numpy as np
import re
from ast import literal_eval
from collections import Counter
from difflib import SequenceMatcher
from metaphone import doublemetaphone
def run_entity_resolution(input_csv,name_col="Complainant",sep=";"):
    # -------------------------------
    # 1. Import necessary libraries
    # -------------------------------
    

    # For vector-based similarity.
    # from sentence_transformers import SentenceTransformer
    # import torch
    # import torch.nn.functional as F

    # -------------------------------
    # 2. Initialize the SentenceTransformer model
    # -------------------------------
    # We use a lightweight, pre-trained model.
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # def get_embedding(name):
    #     """
    #     Compute and return the embedding vector for a name.
    #     """
    #     return model.encode(name)

    # def cosine_similarity(vec1, vec2):
    #     """
    #     Compute cosine similarity between two vectors.
    #     """
    #     t1 = torch.tensor(vec1)
    #     t2 = torch.tensor(vec2)
    #     return F.cosine_similarity(t1, t2, dim=0).item()

    # -------------------------------
    # 3. Load the dataset
    # -------------------------------
    df = pd.read_csv(input_csv,sep=sep)

    # -------------------------------
    # 4. Parse the 'Resolved Entities' column and extract person-type entities
    # -------------------------------
    # def parse_resolved_entities(resolved_entities_str):
    #     try:
    #         return literal_eval(resolved_entities_str)
    #     except Exception:
    #         return []

    # df['Parsed_Entities'] = df['Resolved Entities'].apply(parse_resolved_entities)

    # def extract_person_entities(entities):
    #     # Only extract entities whose type is one of these:
    #     return [entity[0] for entity in entities if entity[1] in ['PERSON', 'POL', 'DIR']]

    # df['Person_Entities'] = df['Parsed_Entities'].apply(extract_person_entities)

    # Collect all unique person entities from all articles.
    all_entities = df[name_col].explode().dropna().tolist()

    # -------------------------------
    # Manual corrections for popular politicians.
    # -------------------------------
    #manula_correction={}

    # def apply_manual_correction(name):
    #     """
    #     Apply manual correction to a preprocessed name.
    #     For names containing 'alias', take only the part before the alias.
    #     Then, if the name is in the corrections dict, return the canonical name.
    #     """
    #     if "alias" in name:
    #         name = name.split("alias")[0].strip()
    #     return manual_corrections.get(name, name)

    # -------------------------------
    # 5. Preprocess entity names with manual cleaning rules
    # -------------------------------
    def preprocess_name(name):
        """
        Preprocess a name by:
        - Removing possessives (e.g. 's and Unicode variants)
        - Removing non-alphabet characters (after handling possessives)
        - Lowercasing and trimming
        - Removing common titles and honorifics
        """
        # Remove possessives (e.g., "'s" or "’s")
        name = re.sub(r"['’]s", "", name)
        # Remove any remaining non-alphabetical characters (retain spaces)
        name = re.sub(r"[^a-zA-Z\s]", "", name)
        # Convert to lowercase and trim whitespace.
        name = name.lower().strip()
        # Remove common titles/honorifics.
        honorifics = {"mr", "mrs", "ms", "dr", "doc", "captain", "sir", "shri","sh.","dr.","shri.","mr.","mrs.","ms."}
        tokens = name.split()
        # Also remove tokens that end with -ji (e.g. "modiji" -> "modi")
        tokens = [token[:-2] if token.endswith("ji") and len(token) > 2 else token for token in tokens]
        tokens = [token for token in tokens if token not in honorifics]
        return " ".join(tokens)

    #all_entities = [apply_manual_correction(preprocess_name(name)) for name in all_entities]
    all_entities = [preprocess_name(name) for name in all_entities]

    # -------------------------------
    # 6. Define abbreviation and similarity functions (rule-based)
    # -------------------------------
    def get_abbreviation(name):
        """
        Extract the first letter of each token, sort them, and return the abbreviation.
        For example:
        'Vidur Kumar Kaushik' --> tokens: ['vidur', 'kumar', 'kaushik']
                                --> initials: ['v', 'k', 'k'] --> sorted: ['k', 'k', 'v'] --> 'KKV'
        """
        # Remove non-alphabet characters just in case.
        name = re.sub(r'[^a-zA-Z\s]', '', name)
        tokens = name.lower().split()
        initials = [t[0] for t in tokens if t]
        initials = sorted(initials)
        return ''.join(letter.upper() for letter in initials)

    def abbreviation_subset_check(name1, name2):
        """
        Checks whether the initials (as a multiset) of one name is the same or a subset of the other.
        For example, for 'Vidur Kumar' (VK) and 'Vidur Kumar Kaushik' (VKK),
        Counter('VK') is a subset of Counter('VKK').
        """
        abbr1 = get_abbreviation(name1)
        abbr2 = get_abbreviation(name2)
        c1 = Counter(abbr1)
        c2 = Counter(abbr2)
        return (c1 <= c2) or (c2 <= c1)

    def levenshtein_ratio(s1, s2):
        return SequenceMatcher(None, s1, s2).ratio()

    def metaphone_match(name1, name2):
        meta1 = doublemetaphone(name1)
        meta2 = doublemetaphone(name2)
        # Return True if any of the non-empty metaphone codes match.
        return any(m1 == m2 for m1 in meta1 for m2 in meta2 if m1 and m2)

    # -------------------------------
    # 7. Pre-compute vector embeddings for all entities
    # -------------------------------
    # entity_embeddings = {name: get_embedding(name) for name in all_entities}

    # -------------------------------
    # 8. Define the matching function (combining rule-based and vector-based support)
    # -------------------------------
    def check_match(name1, name2, vector_threshold=0.6):
        """
        Determines whether two names should be clustered together.
        Combines rule-based checks (abbreviation subset, token count, fuzzy last name check,
        first name Levenshtein ratio, and metaphone match) with vector-based similarity support.
        """
        # First, enforce that the initials (abbreviation) of one must be the same or a subset of the other.
        if not abbreviation_subset_check(name1, name2):
            return False

        # Tokenize names (remove periods, lowercase).
        name1_words = name1.replace('.', '').lower().split()
        name2_words = name2.replace('.', '').lower().split()

        # Skip if either name has fewer than 2 tokens.
        if len(name1_words) < 2 or len(name2_words) < 2:
            return False

        # Fuzzy last name check (using Levenshtein ratio).
        # Instead of an exact match, we allow a slight variation.
        if levenshtein_ratio(name1_words[-1], name2_words[-1]) < 0.8:
            return False

        # First name similarity check using Levenshtein ratio.
        if levenshtein_ratio(name1_words[0], name2_words[0]) < 0.75:
            return False

        # Full-name metaphone check.
        if not metaphone_match(' '.join(name1_words), ' '.join(name2_words)):
            return False

        # -------------------------------
        # Vector-based support: Compute cosine similarity.
        # -------------------------------
        # emb1 = entity_embeddings.get(name1, get_embedding(name1))
        # emb2 = entity_embeddings.get(name2, get_embedding(name2))
        # vec_sim = cosine_similarity(emb1, emb2)
        # print(f"Vector similarity between '{name1}' and '{name2}': {vec_sim:.2f}")

        # if vec_sim < vector_threshold:
        #     print(f"Warning: Low vector similarity ({vec_sim:.2f}) for '{name1}' and '{name2}'.")

        return True

    # -------------------------------
    # 9. Build rule-based clusters
    # -------------------------------
    clusters = []
    matched=[]
    unclustered_entities = set(all_entities)

    while unclustered_entities:
        base_entity = unclustered_entities.pop()
        cluster = [base_entity]
        # Compare against a copy of the unclustered set.
        to_check = set(unclustered_entities)
        for other_entity in to_check:
            if check_match(base_entity, other_entity, vector_threshold=0.8):
                print("Matched: ",base_entity,other_entity)
                cluster.append(other_entity)
                unclustered_entities.remove(other_entity)
        clusters.append(cluster)
        print("CLUSTER FORMED - ", cluster)

    #print("\nRULE-BASED CLUSTERS:")
    #for i, cluster in enumerate(clusters):
        #print(f"Cluster {i+1}: {cluster}")

    return clusters
    '''# -------------------------------
    # 10. Define popular names (preprocessed) that should be canonical if present.
    # -------------------------------
    popular_names_set = {
        "narendra modi", "rahul gandhi", "amit shah",
        "shashi tharoor", "kejriwal", "devendra fadnavis"
    }

    # -------------------------------
    # 11. Select canonical names for each cluster
    # -------------------------------
    def choose_canonical(cluster):
        """
        Choose the canonical name from a cluster:
        - If any alias in the cluster is a popular name, return that.
        - Otherwise, select the name with the greatest length.
        """
        for name in cluster:
            if name in popular_names_set:
                return name
        # Fallback: choose the name with the greatest length.
        canonical = cluster[0]
        for name in cluster:
            if len(name) > len(canonical):
                canonical = name
        return canonical

    # Build a mapping from each alias to its canonical name.
    alias_to_canonical = {}
    for cluster in clusters:
        canonical = choose_canonical(cluster)
        for alias in cluster:
            alias_to_canonical[alias] = canonical

    # -------------------------------
    # 12. Resolve entities in the DataFrame using the canonical mapping
    # -------------------------------
    def resolve_entities_with_type(entities, alias_to_canonical):

        popular_single_names = {
            "modi", "tikait", "rihanna", "amarinder", "greta",
            "amit", "sharad", "adani", "indira", "lalu",
            "rahul", "manmohan", "yogi", "jp", "yediyurappa",
            "arvind", "kejriwal", "mahatma", "pawar", "panneerselvam",
            "chautala", "nanak", "banerjee", "tata", "ambani",
            "nadda", "sukhbir", "tomar", "khattar"
        }

        resolved = []
        for entity in entities:
            if isinstance(entity, (list, tuple)) and len(entity) == 2:
                name, typ = entity
                if typ in ['PERSON', 'POL', 'DIR']:
                    name_proc = apply_manual_correction(preprocess_name(name))
                    # Special reclassification: if entity is "rajya sabha", set type to ORG.
                    if name_proc == "rajya sabha":
                        resolved.append((name_proc, "ORG"))
                        continue
                    resolved_name = alias_to_canonical.get(name_proc, name_proc)
                    # Debug: print the processed name and its resolved version.
                    print(f"DEBUG: Processing tuple entity: original='{name}', preprocessed='{name_proc}', resolved='{resolved_name}'")
                    
                    # Remove if resolved name is one of the unwanted names.
                    if resolved_name.strip() in ["singh", "gargi singh", "abhinav saha", "the big story", "big story"]:
                        continue

                    # Remove single-token names unless in allowlist.
                    tokens = resolved_name.split()
                    if len(tokens) == 1 and resolved_name not in popular_single_names:
                        continue

                    resolved.append((resolved_name, typ))
            else:
                # If the entity isn't in tuple format, process as a name.
                name_proc = apply_manual_correction(preprocess_name(entity))
                resolved_name = alias_to_canonical.get(name_proc, name_proc)
                # Debug: print the processed name and its resolved version.
                print(f"DEBUG: Processing non-tuple entity: original='{entity}', preprocessed='{name_proc}', resolved='{resolved_name}'")
                
                if resolved_name.strip() in ["singh", "gargi singh", "abhinav saha", "the big story", "big story"]:
                    continue

                tokens = resolved_name.split()
                if len(tokens) == 1 and resolved_name not in popular_single_names:
                    continue

                resolved.append((resolved_name, None))
        return resolved

    df['Resolved_Person_Entities'] = df['Parsed_Entities'].apply(
        lambda entities: resolve_entities_with_type(entities, alias_to_canonical)
    )

    # -------------------------------
    # 13. Write the results to CSV
    # -------------------------------
    df.to_csv(output_csv, index=False)
    print("\nAliasFinal_with_vector.csv has been saved.")'''
