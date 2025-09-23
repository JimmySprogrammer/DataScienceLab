# -*- coding: utf-8 -*-
"""
宿舍管理程序
系统用户：宿舍管理员
功能：管理学生宿舍信息
"""

# 初始化学生信息列表（使用字典存储每个学生的信息）
students = [
    {"学号": "2021001", "姓名": "张三", "性别": "男", "宿舍房间号": "A101", "联系电话": "13812345678"},
    {"学号": "2021002", "姓名": "李四", "性别": "女", "宿舍房间号": "B205", "联系电话": "13987654321"},
    {"学号": "2021003", "姓名": "王五", "性别": "男", "宿舍房间号": "A102", "联系电话": "13711112222"},
    {"学号": "2021004", "姓名": "赵六", "性别": "女", "宿舍房间号": "B206", "联系电话": "13633334444"}
]

def display_menu():
    """显示系统主菜单"""
    print("\n" + "=" * 40)
    print("         宿舍管理系统")
    print("=" * 40)
    print("1. 按学号查找学生信息")
    print("2. 录入新的学生信息")
    print("3. 显示所有学生信息")
    print("4. 退出系统")
    print("=" * 40)

def find_student():
    """按学号查找学生信息"""
    student_id = input("请输入要查找的学生学号: ").strip()
    
    if not student_id:
        print("错误：学号不能为空！")
        return
    
    found = False
    for student in students:
        if student["学号"] == student_id:
            print("\n找到学生信息：")
            print("-" * 40)
            print(f"学号: {student['学号']}")
            print(f"姓名: {student['姓名']}")
            print(f"性别: {student['性别']}")
            print(f"宿舍房间号: {student['宿舍房间号']}")
            print(f"联系电话: {student['联系电话']}")
            print("-" * 40)
            found = True
            break
    
    if not found:
        print(f"未找到学号为 {student_id} 的学生信息")

def add_student():
    """录入新的学生信息"""
    print("\n请输入新学生信息")
    print("-" * 30)
    
    # 获取输入并验证
    student_id = input("学号: ").strip()
    if not student_id:
        print("错误：学号不能为空！")
        return
    
    # 检查学号是否已存在
    for student in students:
        if student["学号"] == student_id:
            print(f"错误：学号 {student_id} 已存在！")
            return
    
    name = input("姓名: ").strip()
    if not name:
        print("错误：姓名不能为空！")
        return
    
    gender = input("性别(男/女): ").strip()
    if gender not in ["男", "女"]:
        print("错误：性别只能是'男'或'女'！")
        return
    
    room = input("宿舍房间号: ").strip()
    if not room:
        print("错误：宿舍房间号不能为空！")
        return
    
    phone = input("联系电话: ").strip()
    if not phone:
        print("错误：联系电话不能为空！")
        return
    
    # 确认是否添加
    confirm = input("确认添加以上信息吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消添加操作")
        return
    
    # 添加新学生
    new_student = {
        "学号": student_id,
        "姓名": name,
        "性别": gender,
        "宿舍房间号": room,
        "联系电话": phone
    }
    
    students.append(new_student)
    print(f"成功添加学生 {name} 的信息！")

def display_all_students():
    """显示所有学生信息"""
    if not students:
        print("当前没有任何学生信息！")
        return
    
    print("\n所有学生信息：")
    print("=" * 70)
    print(f"{'学号':<10} {'姓名':<8} {'性别':<4} {'宿舍房间号':<10} {'联系电话':<12}")
    print("-" * 70)
    
    for student in students:
        print(f"{student['学号']:<10} {student['姓名']:<8} {student['性别']:<4} "
              f"{student['宿舍房间号']:<10} {student['联系电话']:<12}")
    
    print("=" * 70)
    print(f"总共 {len(students)} 名学生")

def main():
    """主函数，程序入口"""
    print("欢迎使用宿舍管理系统！")
    
    while True:
        display_menu()
        choice = input("请选择操作 (1-4): ").strip()
        
        if choice == "1":
            find_student()
        elif choice == "2":
            add_student()
        elif choice == "3":
            display_all_students()
        elif choice == "4":
            print("感谢使用宿舍管理系统，再见！")
            break
        else:
            print("无效选择，请重新输入！")
        
        # 暂停一下，让用户看到结果
        input("\n按回车键继续...")

# 程序入口
if __name__ == "__main__":
    main()