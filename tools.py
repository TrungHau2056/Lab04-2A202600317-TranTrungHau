from langchain_core.tools import tool

# ==========================================================
# MOCK DATA – Dữ liệu giả lập hệ thống du lịch
# Lưu ý: Giá cả có logic (VD: cuối tuần đắt hơn, hạng cao hơn đắt hơn)
# Sinh viên cần đọc hiểu data để debug test cases.
# ==========================================================

FLIGHTS_DB = {
    ("Hà Nội", "Đà Nẵng"): [
        {"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "07:20", "price": 1_450_000, "class": "economy"},
        {"airline": "Vietnam Airlines", "departure": "14:00", "arrival": "15:20", "price": 2_800_000, "class": "business"},
        {"airline": "VietJet Air", "departure": "08:30", "arrival": "09:50", "price": 890_000, "class": "economy"},
        {"airline": "Bamboo Airways", "departure": "11:00", "arrival": "12:20", "price": 1_200_000, "class": "economy"},
    ],
    ("Hà Nội", "Phú Quốc"): [
        {"airline": "Vietnam Airlines", "departure": "07:00", "arrival": "09:15", "price": 2_100_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "10:00", "arrival": "12:15", "price": 1_350_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "16:00", "arrival": "18:15", "price": 1_100_000, "class": "economy"},
    ],
    ("Hà Nội", "Hồ Chí Minh"): [
        {"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "08:10", "price": 1_600_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "07:30", "arrival": "09:40", "price": 950_000, "class": "economy"},
        {"airline": "Bamboo Airways", "departure": "12:00", "arrival": "14:10", "price": 1_300_000, "class": "economy"},
        {"airline": "Vietnam Airlines", "departure": "18:00", "arrival": "20:10", "price": 3_200_000, "class": "business"},
    ],
    ("Hồ Chí Minh", "Đà Nẵng"): [
        {"airline": "Vietnam Airlines", "departure": "09:00", "arrival": "10:20", "price": 1_300_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "13:00", "arrival": "14:20", "price": 780_000, "class": "economy"},
    ],
    ("Hồ Chí Minh", "Phú Quốc"): [
        {"airline": "Vietnam Airlines", "departure": "08:00", "arrival": "09:00", "price": 1_100_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "15:00", "arrival": "16:00", "price": 650_000, "class": "economy"},
    ]
}

HOTELS_DB = {
    "Đà Nẵng": [
        {"name": "Mường Thanh Luxury", "stars": 5, "price_per_night": 1_800_000, "area": "Mỹ Khê", "rating": 4.5},
        {"name": "Sala Danang Beach", "stars": 4, "price_per_night": 1_200_000, "area": "Mỹ Khê", "rating": 4.3},
        {"name": "Fivitel Danang", "stars": 3, "price_per_night": 650_000, "area": "Sơn Trà", "rating": 4.1},
        {"name": "Memory Hostel", "stars": 2, "price_per_night": 250_000, "area": "Hải Châu", "rating": 4.6},
        {"name": "Christina's Homestay", "stars": 2, "price_per_night": 350_000, "area": "An Thượng", "rating": 4.7},
    ],
    "Phú Quốc": [
        {"name": "Vinpearl Resort", "stars": 5, "price_per_night": 3_500_000, "area": "Bãi Dài", "rating": 4.4},
        {"name": "Sol by Meliá", "stars": 4, "price_per_night": 1_500_000, "area": "Bãi Trường", "rating": 4.2},
        {"name": "Lahana Resort", "stars": 3, "price_per_night": 800_000, "area": "Dương Đông", "rating": 4.0},
        {"name": "9Station Hostel", "stars": 2, "price_per_night": 200_000, "area": "Dương Đông", "rating": 4.5},
    ],
    "Hồ Chí Minh": [
        {"name": "Rex Hotel", "stars": 5, "price_per_night": 2_800_000, "area": "Quận 1", "rating": 4.3},
        {"name": "Liberty Central", "stars": 4, "price_per_night": 1_400_000, "area": "Quận 1", "rating": 4.1},
        {"name": "Cochin Sen Hotel", "stars": 3, "price_per_night": 550_000, "area": "Quận 3", "rating": 4.4},
        {"name": "The Common Room", "stars": 2, "price_per_night": 180_000, "area": "Quận 1", "rating": 4.6},
    ]
}

from langchain_core.tools import tool


@tool
def search_flights(origin: str, destination: str) -> str:
    """
    Tìm kiếm chuyến bay giữa hai thành phố.
    Tham số:
    - origin: thành phố khởi hành (VD: 'Hà Nội', 'Hồ Chí Minh')
    - destination: thành phố đến (VD: 'Đà Nẵng', 'Phú Quốc')
    Trả về danh sách chuyến bay với hãng, giờ bay, giá vé.
    Nếu không tìm thấy chuyến bay, trả về thông báo không có chuyến.
    """
    key = (origin, destination)

    # Tìm trong FLIGHTS_DB
    flights = FLIGHTS_DB.get(key)

    if flights:
        result = f"Các chuyến bay từ {origin} đến {destination}:\n"
        for f in flights:
            price_formatted = f"{f['price']:,}đ".replace(",", ".")
            result += f"- {f['airline']}: {f['departure']} -> {f['arrival']}, Giá: {price_formatted}, Hạng: {f['class']}\n"
        return result

    # Nếu không tìm thấy, thử tìm chiều ngược lại
    reverse_key = (destination, origin)
    if FLIGHTS_DB.get(reverse_key):
        return f"Không tìm thấy chuyến bay từ {origin} đến {destination} (nhưng có chiều ngược lại)."

    return f"Không tìm thấy chuyến bay từ {origin} đến {destination}."


@tool
def search_hotels(city: str, max_price_per_night: int = 99999999) -> str:
    """
    Tìm khách sạn tại một thành phố, có lọc theo giá tối đa mỗi đêm.
    Tham số:
    - city: tên thành phố (VD: 'Đà Nẵng', 'Phú Quốc', 'Hồ Chí Minh')
    - max_price_per_night: giá tối đa mỗi đêm (VNĐ), mặc định không giới hạn.
    Trả về danh sách khách sạn phù hợp với tên, số sao, giá, khu vực, rating.
    """
    hotels = HOTELS_DB.get(city)

    if not hotels:
        return f"Không tìm thấy khách sạn tại {city}."

    # Lọc theo giá và sắp xếp giảm dần theo rating (hoặc theo yêu cầu người dùng)
    filtered_hotels = [h for h in hotels if h['price_per_night'] <= max_price_per_night]

    if not filtered_hotels:
        return f"Không tìm thấy khách sạn tại {city} với giá dưới {max_price_per_night:,}đ/đêm. Hãy thử khoảng giá khác."

    # Sắp xếp theo giá giảm dần
    sorted_hotels = sorted(filtered_hotels, key=lambda x: x['price_per_night'], reverse=True)

    result = f"Các khách sạn tại {city} (giá <= {max_price_per_night:,}đ):\n"
    for h in sorted_hotels:
        price_formatted = f"{h['price_per_night']:,}đ".replace(",", ".")
        result += f"- {h['name']} ({h['stars']}*): {price_formatted}, Khu vực: {h['area']}, Rating: {h['rating']}\n"

    return result


from langchain_core.tools import tool


@tool
def calculate_budget(total_budget: int, expenses: str) -> str:
    """
    Tính toán ngân sách còn lại sau khi trừ các khoản chi phí.
    Tham số:
    - total_budget: tổng ngân sách ban đầu (VNĐ)
    - expenses: chuỗi mô tả các khoản chi, mỗi khoản cách nhau bởi dấu phẩy,
      định dạng 'tên_khoản:số_tiền' (VD: 'vé_máy_bay:890000,khách_sạn:600000')
    Trả về bảng chi tiết các khoản chi và số tiền còn lại.
    """
    try:
        total_spent = 0
        expense_list = expenses.split(',')
        report = "Bảng chi phí:\n"

        for item in expense_list:
            name, amount_str = item.split(':')
            amount = int(amount_str)
            total_spent += amount
            # Format tiền: 1.000.000đ
            formatted_amount = f"{amount:,}đ".replace(",", ".")
            report += f"- {name}: {formatted_amount}\n"

        remaining = total_budget - total_spent

        report += "---\n"
        report += f"Tổng chi: {total_spent:,}đ\n".replace(",", ".")
        report += f"Ngân sách: {total_budget:,}đ\n".replace(",", ".")
        report += f"Còn lại: {remaining:,}đ\n".replace(",", ".")

        if remaining >= 0:
            report += f"Thặng dư ngân sách {remaining:,}đ! Cần đi du lịch."
        else:
            report += f"Thâm hụt ngân sách! Bạn cần {abs(remaining):,}đ."

        return report.replace(",", ".")

    except Exception:
        return "Lỗi: Định dạng expenses không hợp lệ. Vui lòng sử dụng định dạng 'tên_khoản:số_tiền,tên_khoản:số_tiền'."

