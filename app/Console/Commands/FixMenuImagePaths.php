<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Menu;

class FixMenuImagePaths extends Command
{
    protected $signature = 'fix:menu-image-paths';
    protected $description = 'Fix menu image paths to match actual file locations';

    public function handle()
    {
        $this->info('Fixing menu image paths...');
        
        // Map of menu names to correct image paths
        $imageMapping = [
            'Nasi Goreng Spesial' => 'images/menus/nasi-goreng.jpg',
            'Es Teh Manis' => 'images/menus/esteh.jpg', // Note: actual file is esteh.jpg
            'Kentang Goreng' => 'images/menus/kentang-goreng.jpg'
        ];
        
        foreach ($imageMapping as $menuName => $imagePath) {
            $menu = Menu::where('name', $menuName)->first();
            if ($menu) {
                // Check if file actually exists
                if (file_exists(public_path('storage/' . $imagePath))) {
                    $oldPath = $menu->image_path;
                    $menu->update(['image_path' => $imagePath]);
                    $this->info("✅ Updated '{$menuName}': '{$oldPath}' → '{$imagePath}'");
                } else {
                    $this->warn("⚠️  File not found for '{$menuName}': " . $imagePath);
                }
            } else {
                $this->warn("⚠️  Menu not found: {$menuName}");
            }
        }
        
        // Check for any other menus with incorrect paths (missing images/ prefix)
        $menusWithBadPaths = Menu::where('image_path', 'like', 'menus/%')
                                ->where('image_path', 'not like', 'images/menus/%')
                                ->get();
        
        foreach ($menusWithBadPaths as $menu) {
            $currentPath = $menu->image_path;
            $newPath = 'images/' . $currentPath;
            
            if (file_exists(public_path('storage/' . $newPath))) {
                $menu->update(['image_path' => $newPath]);
                $this->info("✅ Fixed path for '{$menu->name}': '{$currentPath}' → '{$newPath}'");
            } else {
                $this->warn("⚠️  Cannot fix '{$menu->name}': file '{$newPath}' doesn't exist");
            }
        }
        
        $this->info('Image path fixing completed!');
        return 0;
    }
}
