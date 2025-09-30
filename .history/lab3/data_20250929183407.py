import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
import logging

class ESISpider:
    def __init__(self, username, password):
        """
        初始化ESI爬虫
        """
        self.base_url = "https://clarivate.com/academia-government/scientific-and-academic-research/research-funding-analytics/essential-science-indicators/"
        self.username = username
        self.password = password
        self.driver = None
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('esi_crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_driver(self):
        """设置浏览器驱动"""
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--headless')  # 无头模式
        chrome_options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
        self.logger.info("浏览器驱动设置完成")
    
    def login(self):
        """登录ESI系统"""
        try:
            self.logger.info("开始登录ESI系统...")
            self.driver.get(self.base_url)
            
            # 等待登录页面加载
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            
            # 输入用户名和密码
            username_input = self.driver.find_element(By.ID, "username")
            password_input = self.driver.find_element(By.ID, "password")
            
            username_input.send_keys(self.username)
            password_input.send_keys(self.password)
            
            # 点击登录按钮
            login_button = self.driver.find_element(By.ID, "login-btn")
            login_button.click()
            
            # 等待登录成功
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, "results-section"))
            )
            
            self.logger.info("登录成功")
            return True
            
        except Exception as e:
            self.logger.error(f"登录失败: {str(e)}")
            return False
    
    def select_institutions(self):
        """选择Institutions结果类型"""
        try:
            self.logger.info("选择Institutions结果类型...")
            
            # 定位Results List下拉框
            results_select = Select(self.driver.find_element(By.ID, "results-type"))
            results_select.select_by_visible_text("Institutions")
            
            # 等待页面刷新
            time.sleep(5)
            self.logger.info("已选择Institutions")
            return True
            
        except Exception as e:
            self.logger.error(f"选择Institutions失败: {str(e)}")
            return False
    
    def get_research_fields(self):
        """获取可用的研究领域列表"""
        try:
            self.logger.info("获取研究领域列表...")
            
            # 点击Filter Results By展开研究领域
            filter_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Filter Results By')]")
            filter_button.click()
            
            time.sleep(2)
            
            # 获取研究领域选项
            research_fields_section = self.driver.find_element(By.ID, "research-fields")
            field_options = research_fields_section.find_elements(By.TAG_NAME, "input")
            
            fields = []
            for option in field_options:
                field_name = option.get_attribute("value")
                if field_name:
                    fields.append(field_name)
            
            self.logger.info(f"找到 {len(fields)} 个研究领域")
            return fields
            
        except Exception as e:
            self.logger.error(f"获取研究领域失败: {str(e)}")
            return []
    
    def download_field_data(self, field_name):
        """下载特定研究领域的数据"""
        try:
            self.logger.info(f"开始下载 {field_name} 的数据...")
            
            # 选择研究领域
            field_checkbox = self.driver.find_element(By.XPATH, f"//input[@value='{field_name}']")
            if not field_checkbox.is_selected():
                field_checkbox.click()
            
            # 等待数据加载
            time.sleep(5)
            
            # 提取表格数据
            table = self.driver.find_element(By.CLASS_NAME, "results-table")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            data = []
            for row in rows[1:]:  # 跳过表头
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 3:
                    row_data = {
                        'rank': cells[0].text,
                        'institution': cells[1].text,
                        'country': cells[2].text,
                        'field': field_name,
                        'indicators': cells[3].text if len(cells) > 3 else ''
                    }
                    data.append(row_data)
            
            # 取消选择当前领域，为下一个做准备
            field_checkbox.click()
            time.sleep(2)
            
            self.logger.info(f"成功获取 {field_name} 的 {len(data)} 条数据")
            return data
            
        except Exception as e:
            self.logger.error(f"下载 {field_name} 数据失败: {str(e)}")
            return []
    
    def download_all_fields_data(self, fields=None):
        """下载所有研究领域的数据"""
        all_data = []
        
        if not fields:
            fields = self.get_research_fields()
        
        # 限制下载的领域数量用于测试
        test_fields = fields[:5]  # 只下载前5个领域进行测试
        
        for field in test_fields:
            field_data = self.download_field_data(field)
            all_data.extend(field_data)
            time.sleep(3)  # 避免请求过于频繁
        
        return all_data
    
    def save_to_excel(self, data, filename="esi_data.xlsx"):
        """保存数据到Excel文件"""
        if not data:
            self.logger.warning("没有数据可保存")
            return
        
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        self.logger.info(f"数据已保存到 {filename}")
        
        return df
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            self.logger.info("浏览器已关闭")

def main():
    # 请替换为你的华师大账号信息
    USERNAME = "your_ecnu_username"  # 替换为实际用户名
    PASSWORD = "your_password"      # 替换为实际密码
    
    # 初始化爬虫
    spider = ESISpider(USERNAME, PASSWORD)
    
    try:
        # 设置驱动并登录
        spider.setup_driver()
        
        if spider.login():
            # 选择Institutions
            if spider.select_institutions():
                # 获取并下载数据
                all_data = spider.download_all_fields_data()
                
                if all_data:
                    # 保存数据
                    df = spider.save_to_excel(all_data)
                    
                    # 进行数据分析
                    analyzer = ESIAnalyzer(df)
                    analyzer.analyze_ecnu_performance()
                    analyzer.generate_report()
                else:
                    spider.logger.error("未能获取到数据")
            else:
                spider.logger.error("选择Institutions失败")
        else:
            spider.logger.error("登录失败")
            
    except Exception as e:
        spider.logger.error(f"程序执行出错: {str(e)}")
    finally:
        spider.close()

class ESIAnalyzer:
    """ESI数据分析类"""
    
    def __init__(self, data):
        self.data = data
        self.ecnu_data = None
        
    def analyze_ecnu_performance(self):
        """分析华东师范大学的学科表现"""
        # 筛选华东师范大学的数据
        self.ecnu_data = self.data[
            self.data['institution'].str.contains('East China Normal University', case=False, na=False) |
            self.data['institution'].str.contains('华东师范大学', na=False)
        ]
        
        if self.ecnu_data.empty:
            print("未找到华东师范大学的相关数据")
            return
        
        print(f"找到华东师范大学在 {len(self.ecnu_data)} 个学科的数据")
        
        # 分析排名情况
        self.ecnu_data['rank_num'] = pd.to_numeric(self.ecnu_data['rank'], errors='coerce')
        
        # 统计排名分布
        rank_stats = self.ecnu_data['rank_num'].describe()
        print("\n华东师范大学学科排名统计:")
        print(f"最佳排名: {rank_stats['min']:.0f}")
        print(f"最差排名: {rank_stats['max']:.0f}")
        print(f"平均排名: {rank_stats['mean']:.1f}")
        print(f"中位数排名: {rank_stats['50%']:.1f}")
        
        # 找出优势学科（排名前100）
        top_100_fields = self.ecnu_data[self.ecnu_data['rank_num'] <= 100]
        if not top_100_fields.empty:
            print(f"\n排名前100的学科 ({len(top_100_fields)} 个):")
            for _, row in top_100_fields.iterrows():
                print(f"  {row['field']}: 第{row['rank_num']:.0f}名")
        
        return self.ecnu_data
    
    def generate_report(self):
        """生成分析报告"""
        if self.ecnu_data is None or self.ecnu_data.empty:
            print("没有华东师范大学的数据可用于生成报告")
            return
        
        report_content = f"""
# 华东师范大学ESI学科分析报告

## 数据概况
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 涉及学科数量: {len(self.ecnu_data)}
- 数据来源: ESI (Essential Science Indicators)

## 学科表现分析

### 排名统计
- 最佳排名: {self.ecnu_data['rank_num'].min():.0f}
- 最差排名: {self.ecnu_data['rank_num'].max():.0f}
- 平均排名: {self.ecnu_data['rank_num'].mean():.1f}
- 中位数排名: {self.ecnu_data['rank_num'].median():.1f}

### 优势学科（排名前100）
"""
        
        top_100_fields = self.ecnu_data[self.ecnu_data['rank_num'] <= 100]
        if not top_100_fields.empty:
            for _, row in top_100_fields.sort_values('rank_num').iterrows():
                report_content += f"- {row['field']}: 全球第{row['rank_num']:.0f}名\n"
        else:
            report_content += "暂无排名前100的学科\n"
        
        # 保存报告
        with open('ECNU_ESI_Analysis_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("分析报告已生成: ECNU_ESI_Analysis_Report.md")
        
        # 同时保存详细数据
        self.ecnu_data.to_excel('ECNU_ESI_Detailed_Data.xlsx', index=False)
        print("详细数据已保存: ECNU_ESI_Detailed_Data.xlsx")

if __name__ == "__main__":
    main()