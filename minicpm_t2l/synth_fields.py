"""Field vocabulary for the synthetic form templates.

Each domain is one kind of document (a hotel registration card, a payslip...).
A field is (key, English label, Chinese label, value kind, kind args). Some
domains also have line items (a table with repeating rows).

The value kinds are implemented in `minicpm_t2l.synth.gen_value`. Values come from
Faker (seeded per document) and a seeded `random.Random`; nothing comes from a
model, so every label is exactly the string that gets printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldDef:
    key: str
    en: str
    zh: str
    kind: str
    args: tuple = ()


@dataclass(frozen=True)
class Domain:
    name: str
    title_en: tuple[str, ...]
    title_zh: tuple[str, ...]
    fields: tuple[FieldDef, ...]
    items: tuple[FieldDef, ...] = ()
    items_key: str = "items"
    items_en: str = "Items"
    items_zh: str = "明细"


F = FieldDef


def _choice(*pairs: tuple[str, str]) -> tuple:
    return tuple(pairs)


_GENDER = _choice(("Male", "男"), ("Female", "女"))
_YESNO = _choice(("Yes", "是"), ("No", "否"))
_PAY = _choice(("Cash", "现金"), ("Credit card", "信用卡"), ("Bank transfer", "银行转账"), ("Alipay", "支付宝"), ("WeChat Pay", "微信支付"))
_ROOM = _choice(("Single", "单人间"), ("Double", "双人间"), ("Twin", "标准间"), ("Suite", "套房"), ("Family", "家庭房"))
_BLOOD = _choice(("A", "A型"), ("B", "B型"), ("AB", "AB型"), ("O", "O型"))
_MARITAL = _choice(("Single", "未婚"), ("Married", "已婚"), ("Divorced", "离异"), ("Widowed", "丧偶"))
_EDU = _choice(("High school", "高中"), ("Bachelor", "本科"), ("Master", "硕士"), ("Doctorate", "博士"), ("Diploma", "大专"))
_FUEL = _choice(("Petrol", "汽油"), ("Diesel", "柴油"), ("Electric", "纯电动"), ("Hybrid", "混合动力"))
_COLOR = _choice(("White", "白色"), ("Black", "黑色"), ("Silver", "银色"), ("Red", "红色"), ("Blue", "蓝色"), ("Grey", "灰色"))
_PLAN = _choice(("Monthly", "月卡"), ("Quarterly", "季卡"), ("Annual", "年卡"), ("Day pass", "次卡"))
_VISA = _choice(("Tourist", "旅游"), ("Business", "商务"), ("Student", "学习"), ("Work", "工作"), ("Transit", "过境"))
_CLAIM = _choice(("Accident", "意外"), ("Illness", "疾病"), ("Theft", "盗窃"), ("Fire", "火灾"), ("Water damage", "水渍"))
_SHIP = _choice(("Standard", "标准快递"), ("Express", "特快"), ("Overnight", "次日达"), ("Economy", "经济型"))
_TICKET = _choice(("General", "普通票"), ("VIP", "贵宾票"), ("Student", "学生票"), ("Early bird", "早鸟票"))
_ACCT = _choice(("Savings", "储蓄账户"), ("Checking", "支票账户"), ("Fixed deposit", "定期存款"))
_GRADE = _choice(("Grade 1", "一年级"), ("Grade 2", "二年级"), ("Grade 3", "三年级"), ("Grade 4", "四年级"), ("Grade 5", "五年级"), ("Grade 6", "六年级"))
_FREQ = _choice(("Once daily", "每日一次"), ("Twice daily", "每日两次"), ("Three times daily", "每日三次"), ("As needed", "必要时"))
_TRANSPORT = _choice(("Flight", "飞机"), ("Train", "火车"), ("Taxi", "出租车"), ("Bus", "大巴"))
_PURPOSE = _choice(("Gift", "礼品"), ("Sample", "样品"), ("Commercial", "商业"), ("Personal use", "自用"))
_UNIT = _choice(("pcs", "件"), ("box", "箱"), ("kg", "千克"), ("set", "套"))
_RESULT_FLAG = _choice(("Normal", "正常"), ("High", "偏高"), ("Low", "偏低"))

# Reusable person fields.
_NAME = F("full_name", "Full name", "姓名", "name")
_PHONE = F("phone", "Phone", "电话", "phone")
_EMAIL = F("email", "Email", "电子邮箱", "email")
_ADDR = F("address", "Address", "地址", "address")
_DOB = F("date_of_birth", "Date of birth", "出生日期", "date", ("dob",))
_IDNO = F("id_number", "ID number", "证件号码", "idcard")
_GENDER_F = F("gender", "Sex", "性别", "choice", _GENDER)

DOMAINS: dict[str, Domain] = {d.name: d for d in [
    Domain("hotel_registration", ("Hotel Registration Card", "Guest Registration Form"), ("酒店入住登记表", "旅客住宿登记单"), (
        F("guest_name", "Guest name", "客人姓名", "name"), _IDNO, _PHONE, _ADDR,
        F("arrival_date", "Arrival date", "入住日期", "date"), F("departure_date", "Departure date", "离店日期", "date"),
        F("room_no", "Room no.", "房号", "int", (101, 2899)), F("room_type", "Room type", "房型", "choice", _ROOM),
        F("nights", "Nights", "入住天数", "int", (1, 14)), F("rate", "Room rate", "房价", "money", (39, 980)),
        F("deposit", "Deposit", "押金", "money", (100, 2000)), F("payment", "Payment method", "付款方式", "choice", _PAY),
        F("nationality", "Nationality", "国籍", "country"), F("guests", "No. of guests", "入住人数", "int", (1, 4)),
    )),
    Domain("patient_intake", ("Patient Intake Form", "New Patient Registration"), ("门诊病人登记表", "患者信息登记表"), (
        F("patient_name", "Patient name", "患者姓名", "name"), _DOB, _GENDER_F, _PHONE, _ADDR,
        F("blood_type", "Blood type", "血型", "choice", _BLOOD), F("allergies", "Allergies", "过敏史", "choice",
          _choice(("None", "无"), ("Penicillin", "青霉素"), ("Peanuts", "花生"), ("Pollen", "花粉"), ("Seafood", "海鲜"))),
        F("emergency_contact", "Emergency contact", "紧急联系人", "name"), F("emergency_phone", "Emergency phone", "紧急联系电话", "phone"),
        F("insurance_no", "Insurance no.", "医保卡号", "code", ("??########",)), F("visit_date", "Visit date", "就诊日期", "date"),
        F("department", "Department", "科室", "choice", _choice(("Cardiology", "心内科"), ("Dermatology", "皮肤科"), ("Pediatrics", "儿科"), ("Orthopedics", "骨科"), ("ENT", "耳鼻喉科"))),
        F("height_cm", "Height (cm)", "身高(cm)", "int", (140, 195)), F("weight_kg", "Weight (kg)", "体重(kg)", "int", (40, 110)),
    )),
    Domain("job_application", ("Job Application Form", "Employment Application"), ("求职申请表", "应聘人员登记表"), (
        F("applicant_name", "Applicant name", "应聘者姓名", "name"), _DOB, _GENDER_F, _PHONE, _EMAIL, _ADDR,
        F("position", "Position applied for", "应聘职位", "job"), F("expected_salary", "Expected salary", "期望薪资", "money", (3000, 40000)),
        F("education", "Highest education", "最高学历", "choice", _EDU), F("school", "School", "毕业院校", "company"),
        F("graduation_date", "Graduation date", "毕业时间", "date"), F("available_from", "Available from", "可到岗日期", "date"),
        F("marital_status", "Marital status", "婚姻状况", "choice", _MARITAL), F("years_experience", "Years of experience", "工作年限", "int", (0, 25)),
    )),
    Domain("shipping_label", ("Shipping Label", "Parcel Waybill"), ("快递面单", "运单"), (
        F("tracking_no", "Tracking no.", "运单号", "code", ("##########",)), F("sender_name", "Sender", "寄件人", "name"),
        F("sender_phone", "Sender phone", "寄件人电话", "phone"), F("sender_address", "Sender address", "寄件地址", "address"),
        F("recipient_name", "Recipient", "收件人", "name"), F("recipient_phone", "Recipient phone", "收件人电话", "phone"),
        F("recipient_address", "Recipient address", "收件地址", "address"), F("weight", "Weight (kg)", "重量(kg)", "dec", (0.2, 30.0, 1)),
        F("service", "Service", "服务类型", "choice", _SHIP), F("ship_date", "Ship date", "寄件日期", "date"),
        F("freight", "Freight", "运费", "money", (6, 260)), F("pieces", "Pieces", "件数", "int", (1, 6)),
    )),
    Domain("bank_account_opening", ("Account Opening Application", "Bank Account Application"), ("银行开户申请书", "个人开户申请表"), (
        _NAME, _IDNO, _DOB, _PHONE, _EMAIL, _ADDR, F("occupation", "Occupation", "职业", "job"),
        F("employer", "Employer", "工作单位", "company"), F("account_type", "Account type", "账户类型", "choice", _ACCT),
        F("initial_deposit", "Initial deposit", "开户金额", "money", (100, 50000)), F("branch", "Branch", "开户网点", "city"),
        F("application_date", "Date", "申请日期", "date"), F("annual_income", "Annual income", "年收入", "money", (10000, 500000)),
    )),
    Domain("insurance_claim", ("Insurance Claim Form", "Claim Notification"), ("保险理赔申请书", "理赔报案单"), (
        F("policy_no", "Policy no.", "保单号", "code", ("P##########",)), F("claimant_name", "Claimant", "申请人", "name"),
        _PHONE, _ADDR, F("incident_date", "Date of incident", "出险日期", "date"), F("incident_place", "Place of incident", "出险地点", "city"),
        F("claim_type", "Type of claim", "理赔类型", "choice", _CLAIM), F("claim_amount", "Amount claimed", "申请金额", "money", (200, 90000)),
        F("bank_account", "Bank account", "收款账号", "code", ("#### #### #### ####",)), F("bank_name", "Bank", "开户银行", "company"),
        F("claim_date", "Date of claim", "申请日期", "date"), F("insured_name", "Insured", "被保险人", "name"),
    )),
    Domain("vehicle_registration", ("Vehicle Registration Form", "Motor Vehicle Record"), ("机动车登记表", "车辆信息登记表"), (
        F("owner_name", "Owner", "车主姓名", "name"), _IDNO, _ADDR, F("plate_no", "Plate no.", "号牌号码", "plate"),
        F("vin", "VIN", "车辆识别代号", "code", ("?????##?#?#######",)), F("engine_no", "Engine no.", "发动机号", "code", ("??#######",)),
        F("make_model", "Make / model", "品牌型号", "choice", _choice(("Toyota Corolla", "丰田卡罗拉"), ("Honda Civic", "本田思域"), ("VW Passat", "大众帕萨特"), ("BYD Han", "比亚迪汉"), ("Ford Focus", "福特福克斯"))),
        F("color", "Colour", "车身颜色", "choice", _COLOR), F("fuel", "Fuel type", "燃料种类", "choice", _FUEL),
        F("registration_date", "Registration date", "注册日期", "date"), F("seats", "Seats", "核定载客", "int", (2, 9)),
        F("mileage", "Odometer (km)", "里程(公里)", "int", (10, 240000)),
    )),
    Domain("school_enrollment", ("Student Enrollment Form", "School Admission Form"), ("学生入学登记表", "新生报名表"), (
        F("student_name", "Student name", "学生姓名", "name"), _DOB, _GENDER_F, F("grade", "Grade", "年级", "choice", _GRADE),
        F("guardian_name", "Parent / guardian", "监护人", "name"), F("guardian_phone", "Guardian phone", "监护人电话", "phone"), _ADDR,
        F("previous_school", "Previous school", "原就读学校", "company"), F("enrollment_date", "Enrollment date", "入学日期", "date"),
        F("student_id", "Student ID", "学号", "code", ("2026####",)), F("class_no", "Class", "班级", "int", (1, 12)),
        F("bus", "School bus", "乘坐校车", "choice", _YESNO),
    )),
    Domain("library_membership", ("Library Card Application", "Library Membership Form"), ("图书馆借书证申请表", "读者登记表"), (
        _NAME, _IDNO, _PHONE, _EMAIL, _ADDR, F("card_no", "Card no.", "借书证号", "code", ("L#######",)),
        F("member_type", "Member type", "读者类型", "choice", _choice(("Adult", "成人"), ("Child", "少儿"), ("Senior", "老年"), ("Student", "学生"))),
        F("issue_date", "Issue date", "办证日期", "date"), F("expiry_date", "Expiry date", "有效期至", "date"),
        F("deposit", "Deposit", "押金", "money", (20, 200)), F("branch", "Branch", "分馆", "city"),
    )),
    Domain("gym_membership", ("Gym Membership Agreement", "Fitness Club Sign-up"), ("健身会员入会表", "会员卡办理单"), (
        F("member_name", "Member name", "会员姓名", "name"), _PHONE, _DOB, _GENDER_F, F("member_no", "Member no.", "会员编号", "code", ("M######",)),
        F("plan", "Plan", "卡种", "choice", _PLAN), F("start_date", "Start date", "开卡日期", "date"), F("end_date", "End date", "到期日期", "date"),
        F("fee", "Fee", "费用", "money", (99, 6800)), F("trainer", "Personal trainer", "私教", "name"),
        F("locker", "Locker no.", "储物柜号", "int", (1, 400)), F("payment", "Payment", "付款方式", "choice", _PAY),
    )),
    Domain("rental_application", ("Rental Application", "Tenancy Application Form"), ("租房申请表", "房屋租赁登记表"), (
        F("tenant_name", "Applicant", "承租人", "name"), _IDNO, _PHONE, _EMAIL, F("property_address", "Property address", "房屋地址", "address"),
        F("monthly_rent", "Monthly rent", "月租金", "money", (800, 12000)), F("deposit", "Security deposit", "押金", "money", (800, 24000)),
        F("lease_start", "Lease start", "起租日期", "date"), F("lease_end", "Lease end", "到期日期", "date"),
        F("occupants", "Occupants", "居住人数", "int", (1, 6)), F("employer", "Employer", "工作单位", "company"),
        F("landlord_name", "Landlord", "出租人", "name"), F("pets", "Pets", "是否养宠物", "choice", _YESNO),
    )),
    Domain("purchase_order", ("Purchase Order", "PO"), ("采购订单", "订货单"), (
        F("po_number", "PO number", "订单编号", "code", ("PO-######",)), F("order_date", "Order date", "下单日期", "date"),
        F("vendor", "Vendor", "供应商", "company"), F("vendor_address", "Vendor address", "供应商地址", "address"),
        F("ship_to", "Ship to", "收货地址", "address"), F("buyer", "Buyer", "采购员", "name"), F("delivery_date", "Delivery date", "交货日期", "date"),
        F("total", "Total", "合计金额", "money", (100, 99000)), F("payment_terms", "Payment terms", "付款条件", "choice",
          _choice(("Net 30", "月结30天"), ("Net 60", "月结60天"), ("Prepaid", "预付"), ("COD", "货到付款"))),
    ), items=(F("description", "Description", "品名", "product"), F("qty", "Qty", "数量", "int", (1, 500)),
              F("unit", "Unit", "单位", "choice", _UNIT), F("unit_price", "Unit price", "单价", "money", (1, 900)),
              F("amount", "Amount", "金额", "money", (10, 9000)))),
    Domain("utility_bill", ("Electricity Bill", "Utility Statement"), ("电费缴费通知单", "水电费账单"), (
        F("account_no", "Account no.", "户号", "code", ("##########",)), F("customer_name", "Customer", "户名", "name"), _ADDR,
        F("billing_period", "Billing period", "账期", "period"), F("previous_reading", "Previous reading", "上期读数", "int", (1000, 50000)),
        F("current_reading", "Current reading", "本期读数", "int", (1000, 50000)), F("usage", "Usage (kWh)", "用电量(度)", "int", (20, 2000)),
        F("amount_due", "Amount due", "应缴金额", "money", (15, 1500)), F("due_date", "Due date", "缴费截止日", "date"),
        F("meter_no", "Meter no.", "电表编号", "code", ("##########",)),
    ), items=(F("charge", "Charge", "收费项目", "choice", _choice(("Energy", "电度电费"), ("Service fee", "服务费"), ("Tax", "税费"), ("Late fee", "滞纳金"), ("Water", "水费"))),
              F("amount", "Amount", "金额", "money", (1, 600))), items_key="charges", items_en="Charges", items_zh="费用明细"),
    Domain("payslip", ("Payslip", "Salary Statement"), ("工资条", "薪资明细单"), (
        F("employee_name", "Employee", "员工姓名", "name"), F("employee_id", "Employee ID", "工号", "code", ("E#####",)),
        F("department", "Department", "部门", "choice", _choice(("Sales", "销售部"), ("Finance", "财务部"), ("R&D", "研发部"), ("HR", "人事部"), ("Operations", "运营部"))),
        F("pay_period", "Pay period", "工资月份", "period"), F("pay_date", "Pay date", "发放日期", "date"),
        F("gross_pay", "Gross pay", "应发工资", "money", (3000, 50000)), F("deductions", "Total deductions", "扣款合计", "money", (100, 9000)),
        F("net_pay", "Net pay", "实发工资", "money", (2500, 45000)), F("bank_account", "Bank account", "工资卡号", "code", ("#### #### #### ###",)),
        F("tax", "Income tax", "个人所得税", "money", (0, 8000)),
    ), items=(F("item", "Item", "项目", "choice", _choice(("Basic salary", "基本工资"), ("Overtime", "加班费"), ("Bonus", "奖金"), ("Allowance", "津贴"), ("Pension", "养老保险"), ("Housing fund", "住房公积金"))),
              F("amount", "Amount", "金额", "money", (50, 20000))), items_key="lines", items_en="Earnings and deductions", items_zh="工资项目"),
    Domain("visa_application", ("Visa Application Form", "Application for Entry Visa"), ("签证申请表", "入境签证申请表"), (
        F("surname", "Surname", "姓", "last_name"), F("given_names", "Given names", "名", "first_name"), _DOB, _GENDER_F,
        F("passport_no", "Passport no.", "护照号码", "code", ("E########",)), F("nationality", "Nationality", "国籍", "country"),
        F("visa_type", "Visa type", "签证种类", "choice", _VISA), F("entry_date", "Intended date of entry", "预计入境日期", "date"),
        F("stay_days", "Length of stay (days)", "停留天数", "int", (3, 180)), F("occupation", "Occupation", "职业", "job"),
        F("host_address", "Address during stay", "在华住址", "address"), _PHONE, F("passport_expiry", "Passport expiry", "护照有效期", "date"),
    )),
    Domain("event_registration", ("Event Registration", "Conference Registration Form"), ("活动报名表", "会议注册表"), (
        F("attendee_name", "Attendee", "参会人", "name"), F("organization", "Organization", "单位", "company"), _EMAIL, _PHONE,
        F("event_date", "Event date", "活动日期", "date"), F("ticket_type", "Ticket type", "票种", "choice", _TICKET),
        F("fee", "Registration fee", "报名费", "money", (0, 3000)), F("badge_no", "Badge no.", "胸牌号", "code", ("B####",)),
        F("dietary", "Dietary needs", "饮食要求", "choice", _choice(("None", "无"), ("Vegetarian", "素食"), ("Halal", "清真"), ("Gluten-free", "无麸质"))),
        F("session", "Session", "分会场", "choice", _choice(("Morning", "上午场"), ("Afternoon", "下午场"), ("Evening", "晚场"))),
        F("job_title", "Job title", "职务", "job"),
    )),
    Domain("lab_report", ("Laboratory Report", "Blood Test Results"), ("检验报告单", "血液检查报告"), (
        F("patient_name", "Patient", "姓名", "name"), _GENDER_F, F("age", "Age", "年龄", "int", (1, 95)),
        F("sample_no", "Sample no.", "标本号", "code", ("S########",)), F("collected", "Collected", "采样日期", "date"),
        F("reported", "Reported", "报告日期", "date"), F("doctor", "Requesting doctor", "送检医生", "name"),
        F("department", "Department", "科室", "choice", _choice(("Internal medicine", "内科"), ("Surgery", "外科"), ("Outpatient", "门诊"), ("Emergency", "急诊"))),
        F("bed_no", "Bed", "床号", "int", (1, 60)),
    ), items=(F("test", "Test", "检验项目", "choice", _choice(("Hemoglobin", "血红蛋白"), ("White cells", "白细胞"), ("Platelets", "血小板"), ("Glucose", "葡萄糖"), ("Cholesterol", "总胆固醇"), ("Creatinine", "肌酐"))),
              F("result", "Result", "结果", "dec", (0.5, 300.0, 1)), F("unit", "Unit", "单位", "choice", _choice(("g/L", "g/L"), ("mmol/L", "mmol/L"), ("10^9/L", "10^9/L"), ("umol/L", "umol/L"))),
              F("flag", "Flag", "提示", "choice", _RESULT_FLAG)), items_key="results", items_en="Results", items_zh="检验结果"),
    Domain("prescription", ("Prescription", "Medication Order"), ("处方笺", "门诊处方"), (
        F("patient_name", "Patient", "姓名", "name"), F("age", "Age", "年龄", "int", (1, 95)), _GENDER_F,
        F("rx_no", "Rx no.", "处方编号", "code", ("RX######",)), F("rx_date", "Date", "开具日期", "date"),
        F("doctor", "Prescriber", "医师", "name"), F("diagnosis", "Diagnosis", "临床诊断", "choice",
          _choice(("Hypertension", "高血压"), ("Bronchitis", "支气管炎"), ("Gastritis", "胃炎"), ("Migraine", "偏头痛"), ("Diabetes", "糖尿病"))),
        F("fee", "Fee", "药费", "money", (5, 900)), F("pharmacist", "Pharmacist", "药师", "name"),
    ), items=(F("drug", "Drug", "药品名称", "choice", _choice(("Amoxicillin", "阿莫西林"), ("Ibuprofen", "布洛芬"), ("Metformin", "二甲双胍"), ("Omeprazole", "奥美拉唑"), ("Amlodipine", "氨氯地平"), ("Loratadine", "氯雷他定"))),
              F("dose", "Dose", "剂量", "choice", _choice(("250 mg", "250mg"), ("500 mg", "500mg"), ("10 mg", "10mg"), ("20 mg", "20mg"))),
              F("frequency", "Frequency", "用法", "choice", _FREQ), F("qty", "Qty", "数量", "int", (1, 60))), items_key="drugs", items_en="Medications", items_zh="药品"),
    Domain("customs_declaration", ("Customs Declaration", "Declaration Form"), ("海关申报单", "进境物品申报单"), (
        F("traveller_name", "Name", "姓名", "name"), F("passport_no", "Passport no.", "护照号码", "code", ("G########",)),
        F("nationality", "Nationality", "国籍", "country"), F("flight_no", "Flight no.", "航班号", "code", ("??####",)),
        F("arrival_date", "Date of arrival", "入境日期", "date"), F("from_country", "Arriving from", "启运国", "country"),
        F("currency_amount", "Cash carried", "携带现金", "money", (0, 20000)), F("purpose", "Purpose of goods", "物品用途", "choice", _PURPOSE),
        F("declared_value", "Total value", "申报总值", "money", (10, 50000)), F("signature_date", "Date signed", "签名日期", "date"),
    ), items=(F("goods", "Goods", "物品名称", "product"), F("qty", "Qty", "数量", "int", (1, 20)), F("value", "Value", "价值", "money", (5, 9000))), items_key="goods", items_en="Goods declared", items_zh="申报物品"),
    Domain("warranty_card", ("Warranty Card", "Product Warranty Registration"), ("保修卡", "产品保修登记卡"), (
        F("product", "Product", "产品名称", "product"), F("model_no", "Model no.", "型号", "code", ("??-####",)),
        F("serial_no", "Serial no.", "序列号", "code", ("SN##########",)), F("purchase_date", "Date of purchase", "购买日期", "date"),
        F("dealer", "Dealer", "经销商", "company"), F("customer_name", "Customer", "用户姓名", "name"), _PHONE, _ADDR,
        F("warranty_months", "Warranty (months)", "保修期(月)", "int", (6, 60)), F("invoice_no", "Invoice no.", "发票号码", "code", ("########",)),
    )),
    Domain("donation_receipt", ("Donation Receipt", "Acknowledgement of Gift"), ("捐赠收据", "公益捐赠票据"), (
        F("receipt_no", "Receipt no.", "票据号码", "code", ("D########",)), F("donor_name", "Donor", "捐赠人", "name"), _ADDR,
        F("donation_date", "Date", "捐赠日期", "date"), F("amount", "Amount", "捐赠金额", "money", (10, 100000)),
        F("method", "Method", "捐赠方式", "choice", _PAY), F("organization", "Received by", "受赠单位", "company"),
        F("project", "Project", "捐赠项目", "choice", _choice(("Education", "助学"), ("Disaster relief", "救灾"), ("Medical aid", "医疗救助"), ("Environment", "环保"))),
        F("handler", "Handled by", "经办人", "name"), _PHONE,
    )),
    Domain("delivery_note", ("Delivery Note", "Goods Received Note"), ("送货单", "出库单"), (
        F("note_no", "Delivery no.", "单号", "code", ("DN######",)), F("delivery_date", "Date", "送货日期", "date"),
        F("customer", "Customer", "客户名称", "company"), F("delivery_address", "Deliver to", "送货地址", "address"),
        F("contact", "Contact", "联系人", "name"), _PHONE, F("driver", "Driver", "司机", "name"), F("vehicle", "Vehicle", "车牌号", "plate"),
        F("order_ref", "Order ref.", "订单号", "code", ("SO-#####",)),
    ), items=(F("item", "Item", "货品名称", "product"), F("qty", "Qty", "数量", "int", (1, 300)), F("unit", "Unit", "单位", "choice", _UNIT),
              F("remark", "Remark", "备注", "choice", _choice(("", ""), ("Fragile", "易碎"), ("Keep dry", "防潮"), ("Urgent", "加急"))))),
    Domain("restaurant_bill", ("Guest Check", "Table Bill"), ("餐厅结账单", "消费账单"), (
        F("restaurant", "Restaurant", "餐厅名称", "company"), F("table_no", "Table", "桌号", "int", (1, 60)), F("guests", "Guests", "人数", "int", (1, 12)),
        F("server", "Server", "服务员", "first_name"), F("bill_date", "Date", "日期", "date"), F("bill_time", "Time", "时间", "time"),
        F("subtotal", "Subtotal", "小计", "money", (20, 3000)), F("service_charge", "Service charge", "服务费", "money", (0, 300)),
        F("total", "Total", "合计", "money", (20, 3300)), F("payment", "Paid by", "支付方式", "choice", _PAY), F("check_no", "Check no.", "单号", "code", ("#######",)),
    ), items=(F("dish", "Item", "菜品", "dish"), F("qty", "Qty", "数量", "int", (1, 6)), F("price", "Price", "金额", "money", (8, 480)))),
    Domain("travel_expense_claim", ("Travel Expense Claim", "Expense Reimbursement Form"), ("差旅费报销单", "费用报销单"), (
        F("employee_name", "Employee", "报销人", "name"), F("employee_id", "Employee ID", "工号", "code", ("E#####",)),
        F("department", "Department", "部门", "choice", _choice(("Sales", "销售部"), ("Finance", "财务部"), ("R&D", "研发部"), ("Marketing", "市场部"))),
        F("trip_purpose", "Purpose of trip", "出差事由", "choice", _choice(("Client visit", "拜访客户"), ("Conference", "参加会议"), ("Training", "培训"), ("Site inspection", "现场检查"))),
        F("destination", "Destination", "出差地点", "city"), F("depart_date", "Departure", "出发日期", "date"), F("return_date", "Return", "返回日期", "date"),
        F("total_claimed", "Total claimed", "报销总额", "money", (50, 30000)), F("advance", "Advance received", "预借金额", "money", (0, 10000)),
        F("approver", "Approved by", "审批人", "name"), F("claim_date", "Date of claim", "报销日期", "date"),
    ), items=(F("date", "Date", "日期", "date"), F("category", "Category", "类别", "choice", _TRANSPORT + _choice(("Hotel", "住宿"), ("Meals", "餐费"))),
              F("amount", "Amount", "金额", "money", (5, 5000))), items_key="expenses", items_en="Expenses", items_zh="费用明细"),
]}

assert len(DOMAINS) == 24, len(DOMAINS)
for _d in DOMAINS.values():
    _keys = [f.key for f in _d.fields]
    assert len(_keys) == len(set(_keys)), _d.name
    assert len(_d.fields) >= 9, _d.name
