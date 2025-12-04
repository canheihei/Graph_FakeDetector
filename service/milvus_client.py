from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.create_connection("default", host="http://192.168.125.128:19530", port="19530")
# 向量维度
DIM = 768

# 创建两个 collection
def create_collection(name):
    """
        在Milvus数据库中创建一个新的集合

        Args:
            name (str): 要创建的集合名称，如"part_nodes"、"relation_nodes"

        Returns:
            Collection: 新创建的Milvus集合对象，可用于后续的数据操作

        Description:
            创建一个包含四个字段的Milvus集合：
            - id: INT64类型的主键字段，需要手动提供ID值
            - label: VARCHAR类型的字符串字段，最大长度100字符
            - embedding: FLOAT_VECTOR类型的向量字段，维度为768维
            - metadata: JSON类型的字段，用于存储灵活的元数据信息

            集合的主键字段不自动生成ID，需要插入数据时手动指定。
            该集合适用于存储带有向量嵌入、标签和元数据的节点数据。
        """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]

    schema = CollectionSchema(
        fields=fields,
        description=f"Collection for {name}"
    )

    collection = Collection(name=name, schema=schema)

    return collection

# 创建集合示例
collection_part = create_collection("part_nodes")
collection_relation = create_collection("relation_nodes")