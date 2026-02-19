<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Menu;
use App\Models\Category;

class SetupSampleMenus extends Command
{
    protected $signature = 'setup:sample-menus';
    protected $description = 'Setup sample menus with proper image paths';

    public function handle()
    {
        $this->info('Setting up sample menus with images...');
        
        // First, ensure we have categories
        $healthyCategory = Category::firstOrCreate([
            'name' => 'Healthy Meals'
        ], [
            'description' => 'Nutritious and wholesome meals',
            'image_path' => 'images/categories/healthy-meals.jpg'
        ]);

        $beverageCategory = Category::firstOrCreate([
            'name' => 'Healthy Beverages'
        ], [
            'description' => 'Fresh and healthy drinks',
            'image_path' => 'images/categories/beverages.jpg'
        ]);

        // Sample menus with the actual image files found
        $sampleMenus = [
            [
                'name' => 'Berry Smoothie',
                'description' => 'Fresh berry smoothie packed with antioxidants',
                'nutrition_fact' => 'Rich in vitamin C, fiber, and antioxidants',
                'price' => 45000,
                'stock' => 20,
                'image_path' => 'images/menus/berry_smoothie.jpg',
                'categories_id' => $beverageCategory->id
            ],
            [
                'name' => 'Chia Pudding',
                'description' => 'Creamy chia seed pudding with fresh fruits',
                'nutrition_fact' => 'High in omega-3, fiber, and protein',
                'price' => 35000,
                'stock' => 15,
                'image_path' => 'images/menus/chia_pudding.jpg',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Chicken Quinoa Bowl',
                'description' => 'Grilled chicken with quinoa and vegetables',
                'nutrition_fact' => 'Complete protein, gluten-free, high fiber',
                'price' => 65000,
                'stock' => 12,
                'image_path' => 'images/menus/chicken_quinoa.jpg',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Detox Green Juice',
                'description' => 'Fresh green vegetable juice blend',
                'nutrition_fact' => 'Alkalizing, vitamin-rich, low calories',
                'price' => 38000,
                'stock' => 25,
                'image_path' => 'images/menus/detox_juice.jpg',
                'categories_id' => $beverageCategory->id
            ],
            [
                'name' => 'Green Tea',
                'description' => 'Premium organic green tea',
                'nutrition_fact' => 'Antioxidant-rich, metabolism boosting',
                'price' => 25000,
                'stock' => 30,
                'image_path' => 'images/menus/green_tea.webp',
                'categories_id' => $beverageCategory->id
            ],
            [
                'name' => 'Kale Caesar Salad',
                'description' => 'Fresh kale salad with healthy caesar dressing',
                'nutrition_fact' => 'Iron-rich, vitamin K, low carb',
                'price' => 42000,
                'stock' => 18,
                'image_path' => 'images/menus/kale_salad.jpg',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Spinach Veggie Wrap',
                'description' => 'Whole wheat wrap with fresh spinach and vegetables',
                'nutrition_fact' => 'Fiber-rich, vitamin-packed, balanced meal',
                'price' => 48000,
                'stock' => 14,
                'image_path' => 'images/menus/spinach_wrap.jpg',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Sweet Potato Soup',
                'description' => 'Creamy roasted sweet potato soup',
                'nutrition_fact' => 'Beta-carotene, vitamin A, comfort food',
                'price' => 36000,
                'stock' => 20,
                'image_path' => 'images/menus/sweet_potato_soup.webp',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Rainbow Veggie Bowl',
                'description' => 'Colorful bowl of seasonal vegetables',
                'nutrition_fact' => 'Multivitamin natural source, high fiber',
                'price' => 52000,
                'stock' => 16,
                'image_path' => 'images/menus/veggie_bowl.jpg',
                'categories_id' => $healthyCategory->id
            ],
            [
                'name' => 'Mediterranean Veggie Platter',
                'description' => 'Mediterranean-style vegetable arrangement',
                'nutrition_fact' => 'Heart-healthy, Mediterranean diet',
                'price' => 58000,
                'stock' => 10,
                'image_path' => 'images/menus/veggie_platter.jpg',
                'categories_id' => $healthyCategory->id
            ]
        ];

        foreach ($sampleMenus as $menuData) {
            Menu::updateOrCreate(
                ['name' => $menuData['name']],
                $menuData
            );
        }

        $this->info('Sample menus created successfully!');
        $this->info('Total menus: ' . Menu::count());
        
        return 0;
    }
}
