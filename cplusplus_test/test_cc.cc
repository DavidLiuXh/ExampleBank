#include <cstring>
#include <iostream>

namespace {

const char* kStrData = "abcd";

class MyString {
  public:
    MyString()
    :data_(nullptr)
    ,length_(0) {
      std::cout << "Call to default constructor" << std::endl;
    }

    MyString(const char* data) {
      std::cout << "Call to char* constructor" << std::endl;
      Clone(data);
    }

    MyString(const MyString& my_string) {
      std::cout << "Call to copy constructor" << std::endl;
      Clone(my_string.data_);
    }

    MyString(MyString&& my_string) noexcept {
      std::cout << "Call to move constructor" << std::endl;
      data_ = my_string.data_;
      length_ = my_string.length_;

      my_string.data_ = nullptr;
      my_string.length_ = 0;
    }

    ~MyString() {
      std::cout << "Call to destructor" << std::endl;
    }

    MyString& operator =(const MyString& my_string) {
      std::cout << "Call to operator =" << std::endl;
      Clone(my_string.data_);
      return *this;
    }

    MyString& operator =(MyString&& my_string) {
      std::cout << "Call to move operator =" << std::endl;
      data_ = my_string.data_;
      length_ = my_string.length_;

      my_string.data_ = nullptr;
      my_string.length_ = 0;

      return *this;
    }

    void Show() {
      if (nullptr != data_) {
        std::cout << "MyString => " << data_ << std::endl;
      } else {
        std::cout << "MyString => null" << std::endl;
      }
    }

  private:
    void Clone(const char* data) {
      if (nullptr != data) {
        length_ = strlen(data) + 1;
        data_ = new char[length_];
        std::memcpy(data_, data, strlen(data));
        data_[length_] = '\0';
      }
    }

    void Move(MyString&& my_string) {
      data_ = my_string.data_;
      length_ = my_string.length_;

      my_string.data_ = nullptr;
      my_string.length_ = 0;
    }
  private:
    char* data_;
    size_t length_;
};

MyString MakeMyString(const char* data) {
  MyString my_str(data);
  return my_str;
}

void testRightValue() {
  int i = 0;
  int& lr = i;
  const int& lr2 = 10;

#if 0
  MyString str1(kStrData);
  MyString str2 = str1;

  MyString str3;
  str3 = str1;
#endif

  /*
   * 1. 如果没有禁用返回值优化，不管是否定义了move constructor, 都输出如下：
   * Call to char* constructor
   * MyString => abcd
   * Call to destructor
   * 
   * 2. 如果禁用返回值优化-fno-elide-constructors，输出如下：
   * Call to char* constructor
   * Call to move constructor MakeMyString中的局部变量move constructor到此函数的返回值临时变是中
   * Call to destructor
   * Call to move constructor
   * Call to destructor
   * MyString => abcd
   * Call to destructor
   *
   * 3. 如果没定义move constructor, 则上面调用的move constructor换成copy constructor

  MyString str4(MakeMyString(kStrData));
  str4.Show();
  */

  MyString str5;
  str5 = MakeMyString(kStrData);
  str5.Show();
}

} //namespace 

int main(int argc, char* argv[]) {
  testRightValue();

  return 0;
}
