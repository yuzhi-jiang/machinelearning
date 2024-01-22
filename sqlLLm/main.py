# from vanna.remote import VannaDefault
# vn = VannaDefault(model='chinook', api_key='cabc48f4a3e8484d87bebae862abb708')
# vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
# vn.ask('每个城市的员工数量')
#
# from vanna.flask import VannaFlaskApp
# VannaFlaskApp(vn).run()


from vanna.remote import VannaDefault
vn = VannaDefault(model='thelook', api_key='cabc48f4a3e8484d87bebae862abb708')

# 连接 本地的sqlite 数据库
vn.connect_to_sqlite('D:\Documents\Chinook.sqlite')
vn.add_ddl("asdfsadf")
vn.add_question_sql("What is the average salary of employees","SELECT AVG(salary) FROM employees")
vn.ask('所有的users，状态为1的')

from vanna.flask import VannaFlaskApp
VannaFlaskApp(vn).run()