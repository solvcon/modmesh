#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#include <modmesh/serialization/SerializableItem.hpp>
namespace modmesh
{

namespace detail
{

struct Address : SerializableItem
{
    std::string country;
    std::string city;
    std::vector<std::string> phone_numbers;
    std::vector<int> zip_codes;

    MM_DECL_SERIALIZABLE(
        register_member("country", country);
        register_member("city", city);
        register_member("phone_numbers", phone_numbers);
        register_member("zip_codes", zip_codes);)
}; // end struct Address

struct Pet : SerializableItem
{
    std::string name;
    int age;
    bool is_dog;
    bool is_cat;

    MM_DECL_SERIALIZABLE(
        register_member("name", name);
        register_member("age", age);
        register_member("is_dog", is_dog);
        register_member("is_cat", is_cat);)
}; // end struct Pet

struct Person : SerializableItem
{
    std::string name;
    int age;
    bool is_student;
    Address address;
    std::vector<Pet> pets;

    MM_DECL_SERIALIZABLE(
        register_member("name", name);
        register_member("age", age);
        register_member("is_student", is_student);
        register_member("address", address);
        register_member("pets", pets);)
}; // end struct Person

Pet create_dog()
{
    Pet pet;
    pet.name = "Fluffy";
    pet.age = 3;
    pet.is_dog = true;
    pet.is_cat = false;
    return pet;
}

Pet create_cat()
{
    Pet pet;
    pet.name = "Whiskers";
    pet.age = 8;
    pet.is_dog = false;
    pet.is_cat = true;
    return pet;
}

Address create_address()
{
    Address address;
    address.country = "USA";
    address.city = "New York";
    address.phone_numbers = {"123-456-7890", "098-765-4321"};
    address.zip_codes = {10001, 10002};
    return address;
}

struct SecretItem : SerializableItem
{
private:
    std::string private_info = "private_info";

public:
    std::string public_info = "public_info";

    MM_DECL_SERIALIZABLE(
        /* not expose public_info */
        register_member("private_info", private_info);)
}; // end struct SecretItem

struct EscapeItem : SerializableItem
{
    std::string escape_string = "\"\\/\b\f\n\r\t";

    MM_DECL_SERIALIZABLE(
        register_member("escape_string", escape_string);)
}; // end struct EscapeItem

} // end namespace detail

TEST(Json, serialization_private_member_partial_exposure)
{
    detail::SecretItem secret;
    std::string json = secret.to_json();
    EXPECT_EQ(json, "{\"private_info\":\"private_info\"}");
}

TEST(Json, serialization_escape)
{
    detail::EscapeItem escape;
    std::string json = escape.to_json();
    EXPECT_EQ(json, "{\"escape_string\":\"\\\"\\\\/\\b\\f\\n\\r\\t\"}");
}

TEST(Json, serialization_simple)
{
    auto pet = detail::create_dog();
    std::string json = pet.to_json();
    EXPECT_EQ(json, "{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false}");
}

TEST(Json, serialization_wrong_order)
{
    auto pet = detail::create_dog();
    std::string json = pet.to_json();
    EXPECT_NE(json, "{\"name\":\"Fluffy\",\"age\":3,\"is_cat\":false,\"is_dog\":true\"}"); // cat and dog are swapped
}

TEST(Json, serialization_with_vector)
{
    auto address = detail::create_address();

    std::string json = address.to_json();
    EXPECT_EQ(json, "{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[\"123-456-7890\",\"098-765-4321\"],\"zip_codes\":[10001,10002]}");
}

TEST(Json, serialization_with_object)
{

    detail::Person person;
    person.name = "John Doe";
    person.age = 30;
    person.is_student = true;
    person.address = detail::create_address();
    person.pets.push_back(detail::create_dog());
    person.pets.push_back(detail::create_cat());

    std::string json = person.to_json();

    std::string answer = std::string("{\"name\":\"John Doe\",\"age\":30,\"is_student\":true,") +
                         "\"address\":{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[\"123-456-7890\",\"098-765-4321\"],\"zip_codes\":[10001,10002]}," +
                         "\"pets\":[{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false},{\"name\":\"Whiskers\",\"age\":8,\"is_dog\":false,\"is_cat\":true}]}";
}

} // namespace modmesh