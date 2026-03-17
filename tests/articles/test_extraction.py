"""Tests for articles extraction schema and loader."""

from pyspark.sql.types import StringType

from src.articles.schema import ARTICLES_SCHEMA, load_articles


class TestArticlesSchema:
    """Schema-level validation for articles dataset."""

    def test_article_id_is_string_type(self):
        field = next(f for f in ARTICLES_SCHEMA.fields if f.name == "article_id")
        assert isinstance(field.dataType, StringType)

    def test_schema_has_expected_field_count(self):
        assert len(ARTICLES_SCHEMA.fields) == 25


class TestLoadArticles:
    """CSV loading behavior with explicit schema."""

    SAMPLE_CSV = (
        "article_id,product_code,prod_name,product_type_no,product_type_name,product_group_name,graphical_appearance_no,graphical_appearance_name,colour_group_code,colour_group_name,perceived_colour_value_id,perceived_colour_value_name,perceived_colour_master_id,perceived_colour_master_name,department_no,department_name,index_code,index_name,index_group_no,index_group_name,section_no,section_name,garment_group_no,garment_group_name,detail_desc\n"
        "0108775015,108775,T-shirt,255,T-shirt,Garment Upper body,1010016,Solid,9,Black,4,Dark,5,Black,1676,Jersey Basic,A,Ladieswear,1,Ladieswear,16,Womens Everyday Basics,1002,Jersey Basic,Soft cotton t-shirt\n"
        "0739590027,739590,Dress,265,Dress,Garment Full body,1010016,Solid,11,White,1,Light,9,White,1322,Dresses,D,Divided,3,Divided,53,Young Girl,1010,Dresses Ladies,Short sleeve dress\n"
    )

    def test_load_articles_preserves_leading_zeroes_in_article_id(self, spark, tmp_path):
        csv_file = tmp_path / "articles.csv"
        csv_file.write_text(self.SAMPLE_CSV)

        first_article_id = load_articles(spark, str(csv_file)).select("article_id").first()[0]

        assert first_article_id == "0108775015"
