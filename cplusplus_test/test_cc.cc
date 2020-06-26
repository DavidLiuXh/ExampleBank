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

    MyString& operator =(const MyString& my_string) {
      std::cout << "Call to operator =" << std::endl;
      Clone(my_string.data_);
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
        memcpy(data_, data, strlen(data));
        data_[length_] = '\0';
      }
    }

  private:
    char* data_;
    size_t length_;
};

void testRightValue() {
  int i = 0;
  int& lr = i;
  const int& lr2 = 10;

  MyString str1(kStrData);
  MyString str2 = str1;

  MyString str3;
  str3 = str1;
}
} //namespace 

int main(int argc, char* argv[]) {
  testRightValue();

  return 0;
}
