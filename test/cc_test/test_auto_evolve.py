"""
娴嬭瘯鑷姩杩涘寲鍔熻兘
"""
import sys
import os

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺緞
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from alignment.evolver import graph_evolver, UnmappedFeature
from service.neo_client import neo4j_client


def test_neo4j_connection():
    """娴嬭瘯Neo4j杩炴帴"""
    print("=" * 50)
    print("娴嬭瘯Neo4j杩炴帴")
    print("=" * 50)

    try:
        result = neo4j_client.query("RETURN 1 AS test")
        print("鉁?Neo4j杩炴帴鎴愬姛")
        print(f"娴嬭瘯鏌ヨ缁撴灉: {result}")
        return True
    except Exception as e:
        print(f"鉂?Neo4j杩炴帴澶辫触: {e}")
        return False


def test_view_graph_structure():
    """鏌ョ湅鍥捐氨缁撴瀯"""
    print("\n" + "=" * 50)
    print("鏌ョ湅鍥捐氨缁撴瀯")
    print("=" * 50)

    try:
        # 鏌ヨMainDomain
        main_domains = neo4j_client.query("""
            MATCH (m:MainDomain)
            RETURN m.name AS name, m.describe AS describe
        """)
        print(f"\n涓诲煙 (MainDomain): {len(main_domains)}涓?)
        for md in main_domains:
            print(f"  - {md['name']}: {md.get('describe', '')}")

        # 鏌ヨSpecificDomain
        specific_domains = neo4j_client.query("""
            MATCH (s:SpecificDomain)-[:KINDS_OF]->(m:MainDomain)
            RETURN s.name AS name, s.describe AS describe, m.name AS main_domain
        """)
        print(f"\n鍏蜂綋鍩?(SpecificDomain): {len(specific_domains)}涓?)
        for sd in specific_domains[:5]:  # 鍙樉绀哄墠5涓?
            print(f"  - {sd['name']} (灞炰簬: {sd['main_domain']})")

        # 鏌ヨSubDomain
        sub_domains = neo4j_client.query("""
            MATCH (sub:SubDomain)-[:SPECIFIC_OF]->(s:SpecificDomain)
            RETURN sub.name AS name, s.name AS specific_domain
        """)
        print(f"\n瀛愬煙 (SubDomain): {len(sub_domains)}涓?)
        for sub in sub_domains[:5]:  # 鍙樉绀哄墠5涓?
            print(f"  - {sub['name']} (灞炰簬: {sub['specific_domain']})")

        return True
    except Exception as e:
        print(f"鉂?鏌ヨ鍥捐氨缁撴瀯澶辫触: {e}")
        return False


def test_suggest_domain():
    """娴嬭瘯鍩熸帹鑽愬姛鑳?""
    print("\n" + "=" * 50)
    print("娴嬭瘯鍩熸帹鑽愬姛鑳?)
    print("=" * 50)

    # 鍒涘缓娴嬭瘯鐗瑰緛
    test_feature = UnmappedFeature(
        detector="TestDetector",
        feature="test_feature",
        score=0.85,
        raw_value=0.85
    )

    try:
        suggested = graph_evolver.suggest_domain(test_feature)
        if suggested:
            print(f"鉁?鎺ㄨ崘鍩? {suggested['name']}")
            print(f"   鎻忚堪: {suggested.get('describe', '')}")
            return True
        else:
            print("鈿狅笍  鏈壘鍒版帹鑽愬煙锛堝彲鑳藉浘璋变负绌猴級")
            return False
    except Exception as e:
        print(f"鉂?鍩熸帹鑽愬け璐? {e}")
        return False


def test_auto_evolve():
    """娴嬭瘯鑷姩杩涘寲鍔熻兘"""
    print("\n" + "=" * 50)
    print("娴嬭瘯鑷姩杩涘寲鍔熻兘")
    print("=" * 50)

    # 鍒涘缓娴嬭瘯鐗瑰緛
    test_feature = UnmappedFeature(
        detector="TestDetector",
        feature="test_auto_evolve_feature",
        score=0.75,
        raw_value=0.75
    )

    try:
        result = graph_evolver.auto_evolve(test_feature, update_config=False)
        if result:
            print(f"鉁?鑷姩杩涘寲鎴愬姛")
            print(f"   鐗瑰緛鍚嶇О: {result['name']}")
            print(f"   鐗瑰緛鎻忚堪: {result['describe']}")
            print(f"   鎵€灞炲煙: {result['specific_domain']}")
            print(f"   瀛愬煙ID: {result['sub_id']}")
            return True
        else:
            print("鉂?鑷姩杩涘寲澶辫触锛堝彲鑳藉浘璋变负绌猴級")
            return False
    except Exception as e:
        print(f"鉂?鑷姩杩涘寲澶辫触: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("寮€濮嬫祴璇曡嚜鍔ㄨ繘鍖栧姛鑳絓n")

    # 杩愯娴嬭瘯
    tests = [
        ("Neo4j杩炴帴", test_neo4j_connection),
        ("鍥捐氨缁撴瀯", test_view_graph_structure),
        ("鍩熸帹鑽?, test_suggest_domain),
        ("鑷姩杩涘寲", test_auto_evolve),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n鉂?娴嬭瘯 '{name}' 鍑虹幇寮傚父: {e}")
            results.append((name, False))

    # 鎬荤粨
    print("\n" + "=" * 50)
    print("娴嬭瘯鎬荤粨")
    print("=" * 50)
    for name, success in results:
        status = "鉁?閫氳繃" if success else "鉂?澶辫触"
        print(f"{status} - {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n閫氳繃鐜? {passed}/{total} ({passed/total*100:.1f}%)")
