# from vanna.base import VannaBase
#
#
# class MyCustomVectorDB(VannaBase):
#     def add_ddl(self, ddl: str, **kwargs) -> str:
#
#     # Implement here
#
#     def add_documentation(self, doc: str, **kwargs) -> str:
#
#     # Implement here
#
#     def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
#
#     # Implement here
#
#     def get_related_ddl(self, question: str, **kwargs) -> list:
#
#     # Implement here
#
#     def get_related_documentation(self, question: str, **kwargs) -> list:
#
#     # Implement here
#
#     def get_similar_question_sql(self, question: str, **kwargs) -> list:
#
#     # Implement here
#
#     def get_training_data(self, **kwargs) -> pd.DataFrame:
#
#     # Implement here
#
#     def remove_training_data(id: str, **kwargs) -> bool:
#
#
# # Implement here
#
# class MyCustomLLM(VannaBase):
#     def __init__(self, config=None):
#         pass
#
#     def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
#
#     # Implement here
#
#     def generate_question(self, sql: str, **kwargs) -> str:
#
#     # Implement here
#
#     def get_followup_questions_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list,
#                                       **kwargs):
#
#     # Implement here
#
#     def get_sql_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list, **kwargs):
#
#     # Implement here
#
#     def submit_prompt(self, prompt, **kwargs) -> str:
#
#
# # Implement here
#
#
# class MyVanna(MyCustomVectorDB, MyCustomLLM):
#     def __init__(self, config=None):
#         MyCustomVectorDB.__init__(self, config=config)
#         MyCustomLLM.__init__(self, config=config)
#
#
# vn = MyVanna()
#
#
#
# vn.connect_to_sqlite('my-database.sqlite')
#
# df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
#
# # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
# plan = vn.get_training_plan_generic(df_information_schema)
#
#
# vn.train(ddl="""
#     CREATE TABLE IF NOT EXISTS my-table (
#         id INT PRIMARY KEY,
#         name VARCHAR(100),
#         age INT
#     )
# """)
#
# training_data = vn.get_training_data()
# training_data
#
#
# # vn.ask(question=...)
# #
# # from vanna.flask import VannaFlaskApp
# # app = VannaFlaskApp(vn)
# # app.run()